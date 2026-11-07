"""Random-initialized training; torchrun enables one process per GPU."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset

from .config import RunConfig
from .data.split_manifest import load_manifest, load_window, windows
from .gpu import auto_configure_training, launch_training, loader_workers, resolve_gpus
from .io import write_json
from .logs import get_logger, progress
from .models.lidardm_adapter import LidarDMAdapter
from .provenance import license_gate, provenance, validate_checkpoint
from .data.split_manifest import digest
from .tuning import batch_limit, search_batch


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
    for key in ('labels', 'world_from_sensor'):
        if all(key in sample for sample in samples):
            result[key] = torch.as_tensor(np.stack([sample[key] for sample in samples]))
    return result


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def train(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = config.model_copy(update={'gpus': resolve_gpus(config.gpus)})
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if config.gpus > 1 and not distributed:
        return launch_training(config, output)
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if config.gpus and not torch.cuda.is_available():
        raise ValueError('No CUDA GPU is available; choose gpus=0 for CPU training')
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if config.gpus else 'cpu')
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    torch.set_num_threads(1 if device.type == 'cpu' else max(1, min(4, cpus // (2 * world_size))))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability(device)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    if distributed:
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    batch_size = config.batch_size or 1
    phase = 'setup'
    try:
        torch.manual_seed(config.seed)
        manifest = load_manifest(config.manifest, verify_files=rank == 0)
        license_gate(manifest)
        if rank == 0:
            from .ot.reference import prepare_reference
            config = prepare_reference(config, output, manifest)
            write_json(output / 'request.json', config.model_dump())
        if distributed:
            torch.distributed.barrier()
            config = RunConfig.model_validate_json((output / 'request.json').read_text())
        train_data = RangeDataset(manifest, 'train', config.window)
        validation_data = RangeDataset(manifest, 'validation', config.window)
        sampler = DistributedSampler(train_data, shuffle=True, seed=config.seed) if distributed else None
        resources = auto_configure_training('cpu' if config.gpus == 0 else 'auto', workers=max(1, config.gpus))
        workers = loader_workers(config.workers, 2 * world_size) if device.type == 'cuda' else max(0, config.workers)
        limit, tuning = batch_limit(collate([train_data[0]]), len(sampler) if sampler else len(train_data),
                                    workers, world_size, config.batch_size)
        automatic = config.auto_optimize or config.batch_size == 0
        batch_size = min(limit, 8) if automatic and device.type == 'cpu' else limit
        loader_options = {'num_workers': workers, 'pin_memory': device.type == 'cuda'}
        if workers:
            loader_options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context='spawn')
        # Striding avoids the duplicate padding introduced by DistributedSampler during evaluation.
        validation_shard = Subset(validation_data, range(rank, len(validation_data), world_size))
        range_indices = np.linspace(0, len(validation_data) - 1,
            min(len(validation_data), config.ot.range_validation_windows), dtype=int).tolist()
        resources['checkpoint_selection'] = {'metric': config.ot.checkpoint_metric, 'split': 'validation',
            'window_indices': range_indices if config.ot.checkpoint_metric == 'range_w1_m' else [],
            'sample_steps': config.sample_steps, 'seed_base': config.seed + 27183,
            'output_refinement': config.ot.generation_enabled, 'aggregation': 'Equal weight per supported frame'}
        model_type = LidarDMAdapter
        if any((config.ot.conditional_weight, config.ot.mass_weight, config.ot.physics_weight, config.ot.regularization_weight)):
            from .models.ot_adapter import OTLidarDMAdapter
            model_type = OTLidarDMAdapter
        model = model_type(config.model_dump(), manifest['sensor_meta']).to(device)
        if config.warm_start:
            parent = torch.load(config.warm_start, map_location='cpu', weights_only=True)
            validate_checkpoint(parent, config.manifest)
            model.load_state_dict(parent['state_dict'])
        dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        autocast = lambda: torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
        phase = 'batch_tuning'
        if automatic and device.type == 'cuda':
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            optimizer_bytes = 2 * parameter_bytes
            distributed_bytes = parameter_bytes + 64 * 1024**2 if distributed else 0
            memory_budget = .85 * (free + torch.cuda.memory_allocated(device)) - optimizer_bytes - distributed_bytes
            tuning.update(free_gpu_memory_bytes=free, memory_budget_bytes=memory_budget, optimizer_reserve_bytes=optimizer_bytes,
                          distributed_reserve_bytes=distributed_bytes)
            get_logger().info('Rank %d batch tuning: window=%s, %d available windows, search up to %d, free GPU memory %.2f GiB',
                              rank, tuning['window_shape'], tuning['windows_per_rank'], limit, free / 1024**3)

            def probe(size):
                elapsed, peak = 0., 0
                batch = loss = None
                try:
                    with torch.random.fork_rng(devices=[device.index]):
                        batch = to_device(collate([train_data[index * len(train_data) // size]
                                                   for index in range(size)]), device)
                        for repeat in range(2):
                            model.zero_grad(set_to_none=True)
                            loss = None
                            torch.cuda.reset_peak_memory_stats(device)
                            torch.cuda.synchronize(device)
                            started = time.perf_counter()
                            with autocast():
                                loss = model(batch)['loss']
                            loss.backward()
                            torch.cuda.synchronize(device)
                            if repeat:
                                elapsed = time.perf_counter() - started
                            peak = max(peak, torch.cuda.max_memory_allocated(device))
                    return elapsed, peak
                finally:
                    batch = loss = None
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

            try:
                batch_size, measurements = search_batch(probe, limit, memory_budget)
            except torch.cuda.OutOfMemoryError:
                batch_size, measurements = 0, []
            tuning['measurements'] = measurements
        tuning.update(rank=rank, device=str(device), selected_batch_size=batch_size)
        if distributed:
            shared = torch.tensor(batch_size, device=device)
            torch.distributed.all_reduce(shared, op=torch.distributed.ReduceOp.MIN)
            batch_size = int(shared)
        if batch_size < 1:
            batch_size = 1
            raise torch.cuda.OutOfMemoryError('No safe training batch fits on every assigned GPU')
        if distributed:
            per_rank = [None] * world_size
            torch.distributed.all_gather_object(per_rank, tuning)
        else:
            per_rank = [tuning]
        resources['batch_tuning'] = {**tuning, 'per_rank': per_rank,
                                     'synchronized_batch_size': batch_size, 'automatic': automatic}
        loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                            collate_fn=collate, **loader_options)
        validation = DataLoader(validation_shard, batch_size=batch_size, collate_fn=collate, **loader_options)
        resources.update(world_size=world_size, batch_size_per_process=batch_size,
                         effective_batch_size=batch_size * world_size * config.gradient_accumulation,
                         loader_workers_per_process=workers, validation_windows=len(validation_data))
        if distributed:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                            gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, fused=device.type == 'cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
        history, best = [], float('inf')
        if rank == 0:
            write_json(output / 'resources.json', resources)
        proof = provenance(config.manifest, config.model_dump())
        if config.ot.reference:
            proof['ot_reference_sha256'] = digest(config.ot.reference)
            proof['ot_reference_split'] = 'train'
        if config.warm_start:
            proof['stage_initialization'] = 'own_checkpoint'
            proof['parent_checkpoint_sha256'] = digest(config.warm_start)
        if rank == 0:
            get_logger().info('Training on %d process(es), %d training windows, batch %d per process, effective batch %d',
                              world_size, len(train_data), batch_size, resources['effective_batch_size'])
        get_logger().info('Rank %d/%d uses %s; DataLoader workers=%d, validation windows=%d',
                          rank, world_size, device, workers, len(validation_shard))
        phase = 'training'
        for epoch in range(config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            component_totals, sample_count = {}, 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                update = (step + 1) % config.gradient_accumulation == 0 or step + 1 == len(loader)
                sync = model.no_sync() if distributed and not update else nullcontext()
                with sync:
                    with autocast():
                        components = model(to_device(batch, device))
                        group_start = step // config.gradient_accumulation * config.gradient_accumulation
                        group_size = min(config.gradient_accumulation, len(loader) - group_start)
                        loss = components['loss'] / group_size
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss; checkpoint was not promoted')
                    scaler.scale(loss).backward()
                if update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                sample_count += len(batch['range'])
                for key, value in components.items():
                    component_totals[key] = component_totals.get(key, 0) + value.detach().double() * len(batch['range'])
                if rank == 0 and ((step + 1) % max(1, len(loader) // 20) == 0 or step + 1 == len(loader)):
                    progress(output, f"epoch {epoch + 1}/{config.epochs}, loss={float(components['loss'].detach()):.6f}",
                             step + 1, len(loader))
            raw_model = model.module if distributed else model
            metric = torch.zeros(2, device=device, dtype=torch.float64)
            range_metric = torch.zeros(3, device=device, dtype=torch.float64)
            raw_model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                torch.manual_seed(config.seed + 9182 + rank)
                for batch in validation:
                    with autocast():
                        values = raw_model.train_step(to_device(batch, device))
                    metric[0] += values['loss'].double() * len(batch['range'])
                    metric[1] += len(batch['range'])
                if config.ot.checkpoint_metric == 'range_w1_m':
                    from .ot.range_distance import validate_generated
                    phase = 'range_validation'
                    range_metric = validate_generated(raw_model, validation_data, range_indices[rank::world_size],
                                                      device, config.seed + 27183, autocast)
                    phase = 'training'
            if distributed:
                torch.distributed.all_reduce(metric)
                torch.distributed.all_reduce(range_metric)
            validation_loss = float(metric[0] / metric[1].clamp_min(1))
            validation_range = (float(range_metric[0] / range_metric[1])
                                if range_metric[1] > 0 and range_metric[2] == 0 else None)
            selection_score = validation_loss
            if config.ot.checkpoint_metric == 'range_w1_m':
                selection_score = validation_range if validation_range is not None else float('inf')
            promote = selection_score < best
            if promote:
                best = selection_score
            resources['evaluated_validation_windows'] = int(metric[1])
            names = list(component_totals)
            totals = torch.stack([component_totals[key] for key in names] + [metric.new_tensor(sample_count)])
            if distributed:
                torch.distributed.all_reduce(totals)
            if rank == 0:
                averages = dict(zip(names, (totals[:-1] / totals[-1]).tolist()))
                record = {'epoch': epoch + 1, 'train_loss': averages.pop('loss'), 'validation_loss': validation_loss}
                if config.ot.checkpoint_metric == 'range_w1_m':
                    record.update(validation_range_w1_m=validation_range,
                                  validation_range_frames=int(range_metric[1]),
                                  validation_range_missing_frames=int(range_metric[2]))
                record.update({f'train_{key}': value for key, value in averages.items()})
                history.append(record)
                write_json(output / 'training.json', history)
                get_logger().info('Epoch metrics: %s', json.dumps(record))
                if promote:
                    selection = {**resources['checkpoint_selection'], 'value': best, 'epoch': epoch + 1,
                                 'supported_frames': int(range_metric[1]), 'missing_frames': int(range_metric[2])}
                    checkpoint = {'state_dict': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                                  'provenance': proof, 'sensor_meta': manifest['sensor_meta'],
                                  'config': config.model_dump(), 'epoch': epoch + 1, 'validation_loss': validation_loss,
                                  'validation_range_w1_m': validation_range, 'checkpoint_selection': selection}
                    checkpoint['compute_config'] = resources
                    torch.save(checkpoint, output / 'checkpoint.tmp')
                    (output / 'checkpoint.tmp').replace(output / 'checkpoint.pt')
                    get_logger().info('Checkpoint selected by %s: %.6f at epoch %d',
                                      config.ot.checkpoint_metric, best, epoch + 1)
        if best == float('inf'):
            raise ValueError('No checkpoint has finite validation support for the selected metric; inspect training.json')
        if distributed:
            torch.distributed.barrier()
    except torch.cuda.OutOfMemoryError:
        write_json(Path(output) / f'oom-rank-{rank}.json', {'batch_size': batch_size, 'rank': rank, 'phase': phase})
        raise
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    train(RunConfig.model_validate_json(arguments.request.read_text()), arguments.output)
