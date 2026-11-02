"""Random-initialized training; torchrun enables one process per GPU."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .config import RunConfig
from .data.split_manifest import load_manifest, load_window, windows
from .gpu import auto_configure_training
from .io import write_json
from .logs import get_logger, progress
from .models.lidardm_adapter import LidarDMAdapter
from .provenance import license_gate, provenance


class RangeDataset(Dataset):
    def __init__(self, manifest, split, window):
        self.manifest, self.records = manifest, windows(manifest, split, window)
        if not self.records:
            raise ValueError(f'No complete temporal windows in {split}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sequence_id, records = self.records[index]
        return load_window(records, self.manifest['sensor_meta'], sequence_id)


def collate(samples):
    result = {key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
              for key in ('range', 'valid', 'intensity', 'has_intensity', 'timestamp_ns')}
    result['sequence_id'] = [sample['sequence_id'] for sample in samples]
    result['sensor_meta'] = samples[0]['sensor_meta']
    return result


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def train(config, output):
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0'))
    if config.gpus and not torch.cuda.is_available():
        raise ValueError('No CUDA GPU is available; choose gpus=0 for CPU training')
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if config.gpus else 'cpu')
    torch.set_num_threads(1 if device.type == 'cpu' else 4)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
    if distributed:
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        torch.manual_seed(config.seed)
        manifest = load_manifest(config.manifest, verify_files=True)
        license_gate(manifest)
        train_data = RangeDataset(manifest, 'train', config.window)
        validation_data = RangeDataset(manifest, 'validation', config.window)
        sampler = DistributedSampler(train_data, shuffle=True, seed=config.seed) if distributed else None
        resources = auto_configure_training('cpu' if config.gpus == 0 else 'auto', workers=max(1, config.gpus))
        batch_size = min(config.batch_size, resources['estimated_batch_capacity'])
        loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                            collate_fn=collate, num_workers=config.workers, pin_memory=device.type == 'cuda')
        validation = DataLoader(validation_data, batch_size=config.batch_size, collate_fn=collate)
        model = LidarDMAdapter(config.model_dump(), manifest['sensor_meta']).to(device)
        if distributed:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
        autocast = lambda: torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
        history, best = [], float('inf')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        proof = provenance(config.manifest, config.model_dump())
        if rank == 0:
            get_logger().info('Training on %s with %d worker(s), %d training windows, batch size %d',
                              device, int(os.environ.get('WORLD_SIZE', '1')), len(train_data), batch_size)
        for epoch in range(config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            losses = []
            optimizer.zero_grad()
            for step, batch in enumerate(loader):
                with autocast():
                    components = model(to_device(batch, device))
                    group_start = step // config.gradient_accumulation * config.gradient_accumulation
                    group_size = min(config.gradient_accumulation, len(loader) - group_start)
                    loss = components['loss'] / group_size
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss; checkpoint was not promoted')
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                losses.append(float(components['loss'].detach()))
                if rank == 0 and ((step + 1) % max(1, len(loader) // 20) == 0 or step + 1 == len(loader)):
                    progress(output, f'epoch {epoch + 1}/{config.epochs}, loss={losses[-1]:.6f}', step + 1, len(loader))
            raw_model = model.module if distributed else model
            metric = torch.zeros(2, device=device)
            if rank == 0:
                raw_model.eval()
                with torch.no_grad(), torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                    torch.manual_seed(config.seed + 9182)
                    for batch in validation:
                        values = raw_model.train_step(to_device(batch, device))
                        metric[0] += values['loss'] * len(batch['range'])
                        metric[1] += len(batch['range'])
            if distributed:
                torch.distributed.broadcast(metric, 0)
            validation_loss = float(metric[0] / metric[1].clamp_min(1))
            if rank == 0:
                record = {'epoch': epoch + 1, 'train_loss': sum(losses) / len(losses), 'validation_loss': validation_loss}
                history.append(record)
                write_json(output / 'training.json', history)
                get_logger().info('Epoch metrics: %s', json.dumps(record))
                if validation_loss < best:
                    best = validation_loss
                    checkpoint = {'state_dict': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                                  'provenance': proof, 'sensor_meta': manifest['sensor_meta'],
                                  'config': config.model_dump(), 'epoch': epoch + 1, 'validation_loss': best}
                    checkpoint['compute_config'] = resources
                    torch.save(checkpoint, output / 'checkpoint.tmp')
                    (output / 'checkpoint.tmp').replace(output / 'checkpoint.pt')
        if distributed:
            torch.distributed.barrier()
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    train(RunConfig.model_validate_json(arguments.request.read_text()), arguments.output)
