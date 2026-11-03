# Pandar128 LiDAR generation PoC

Implements `clean_license_lidar_poc_report.pdf` and adds the optional experiment
workflow from `sim_to_real_OT_diffusion_RL_updated_report.pdf`, with **RL retained**.
The Streamlit UI runs on the client. The API, dataset reader, diffusion training,
calibration and evaluation run on the remote server through an SSH tunnel.

The model is a **range-image adaptation of LidarDM's conditional UNet frontend**,
trained from random initialization. It does not reproduce the upstream latent
SDF/VAE world-generation pipeline. No KITTI-360/Waymo weights, pretrained VAEs,
external asset banks or LoRA bases are loaded or downloaded.

## Run

Copy this directory to the remote server. Install a PyTorch build appropriate for
its GPUs, then install `requirements-server.txt` in a separate Python 3.11+ server
environment. Create and activate that environment before installing the GPU build:

```bash
python3 -m venv .venv-server
source .venv-server/bin/activate
# Install your GPU-compatible PyTorch build here, then:
python -m pip install -r requirements-server.txt
python server.py
# Equivalent: uvicorn server:app --host 127.0.0.1 --port 9000 --workers 1 --no-access-log
```

CPU dataset preparation remains independent of Torch/Diffusers. Automatic GPU
preparation imports Torch lazily and falls back to CPU if it is unavailable.
Training and generation load the model stack when needed. This application uses
standard AdamW and its own PPO; PEFT, Transformers and bitsandbytes are not required
server dependencies. If a shared environment raises `HybridCache` import errors or
a missing `libbitsandbytes_cuda130.so`, use the separate environment above rather
than inheriting those incompatible optional packages.

On the client, only the UI dependencies are required:

```bash
python -m pip install -r requirements-client.txt
ssh -i /path/to/your-key.pem YOUR_USER@YOUR_SERVER
# Verify the host key once, then exit that login session.
bash ssh_tunnel.sh start -H YOUR_SERVER -u YOUR_USER -k /path/to/your-key.pem
streamlit run app.py --server.address 127.0.0.1 --server.port 9000
```

Open `http://127.0.0.1:9000`. All application ports are **9000**: Streamlit binds
client `127.0.0.1:9000`, the API tunnel binds client `127.0.0.2:9000`, and the remote
API binds server `127.0.0.1:9000`. Separate client loopback addresses allow the UI
and tunnel to share the same port. The Streamlit defaults are also saved in
`.streamlit/config.toml`. Run these commands from the project directory.

The same tunnel can be opened directly with your working SSH key, instead of the helper:

```bash
ssh -i /path/to/your-key.pem -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.2:9000:127.0.0.1:9000 YOUR_USER@YOUR_SERVER
```

The UI can also start/status/close its own tunnel. For a nondefault SSH port use
`-P 2222`; `-k /path/to/key` selects a client-side key. SSH aliases and agent keys
work. The SSH transport retains its existing port (22 by default); this does not
change the server's SSH daemon. The tunnel forwards between loopback addresses. Closing
the browser or tunnel does not stop remote jobs. Cancellation stops the job's
process group, including its training workers. API shutdown stops active jobs.
Use one Uvicorn worker. `LIDARPOC_RUNS=/data/lidar-runs` changes artifact storage.

## Logs, GPU graphs and existing SSH connections

Worker stdout/stderr is printed immediately in the server terminal and saved to
`runs/<job_id>/worker.log`. **Monitor and results** shows a live processing log and
prints each new chunk in the client's Streamlit terminal. Byte cursors prevent
reprinting history on every refresh. Logs include preparation, training batches,
epoch metrics, generation, calibration evaluations and uncaught tracebacks. The
browser retains the latest 64 KiB; the server retains the complete worker log.
`GET /jobs/{job_id}/logs?offset=0` exposes the same incremental stream.

GPU utilization and used VRAM graphs refresh every two seconds, including while
no job is running. The UI retains up to ten minutes per browser session. A live
table includes GPU names/UUIDs, total VRAM, temperature and power where supported.
These are host-wide physical GPU metrics, not per-job counters. Only the server
needs `nvidia-ml-py`, NVIDIA's maintained NVML bindings; the package is included in
`requirements-server.txt`. CPU/driverless hosts report monitoring unavailability
without blocking jobs. Unsupported fields remain unknown, rather than showing zero.

The UI reuses an API tunnel already responding at its configured endpoint. The
tunnel helper also reuses an existing SSH `ControlMaster` from your SSH
`ControlPath` configuration, or from an explicit `-S` socket path. In the UI,
enter that absolute path in **Existing SSH control socket (optional)**. Closing
the app's forward preserves the shared SSH session. An explicit socket must
already have a running master; the helper reports an error if it is unavailable.

For a connection started with a control socket, a second client terminal can add
the application's forward over the same connection:

```bash
# Client terminal 1: start a reusable login once, or use your existing master
ssh -M -S "$HOME/.ssh/syn-control" -o ControlPersist=600 \
  -i /path/to/your-key.pem YOUR_USER@YOUR_SERVER

# Client terminal 2: attach forwarding to that connection
bash ssh_tunnel.sh start -H YOUR_SERVER -u YOUR_USER -S "$HOME/.ssh/syn-control"

# Remove only the app's forwarding; keep the shared login
bash ssh_tunnel.sh stop
```

A plain `ssh -i` login without a control socket needs a new multiplexed login
before the helper can share it. OpenSSH describes this behavior under
[ControlMaster](https://man.openbsd.org/ssh_config.5#ControlMaster).

## Data and experiments

1. Choose **Prepare Pandar128** and click **Start remote job**. With the dataset
   directory left blank, the server automatically downloads the selected
   [Pandar128](https://huggingface.co/datasets/filipberanek/pandar128-lane-line-detection)
   revision `3366842fa701e4e5ec313f9124acb84b502f11ee` into the server project's
   `datasets/pandar128/` folder. Six archives (about 212 GB compressed) are retained
   in `.downloads/`; extraction writes scans and available `*_odom_list.npz` under
   `train/<trace>/`, `validation/<trace>/`, `test/<trace>/`. Allow additional disk
   space for these extracted records and prepared caches. Downloading and extraction
   run in the remote job, with progress and logs visible in the UI. Hugging Face
   resumes interrupted downloads; `.download.json` tracks fully extracted archives,
   so retries reuse completed work and repair missing/truncated scans. An optional
   directory override accepts an already extracted dataset. `datasets/` is ignored
   by Git.
2. Inspect NPZ schemas from the UI. Published structured scalar XYZ/intensity,
   legacy vector-XYZ/i, and the publisher writer's object-tuple variant are handled.
   Variable-length object NPZ files use NumPy pickle loading; use the publisher's
   files. Preparation holds one source trace in RAM, then writes numeric per-frame
   caches. Source traces may be large even when a small frame limit is selected.
3. Confirm the input coordinate frame. `vehicle` applies the inverse of the
   captured Pandar128 mounting transform; `sensor` uses the points directly.
   Coordinates are metres, sensor axes are right handed, and azimuth is atan2(y,x).
4. **Estimate from training scans** is the default beam setting. Alternatively,
   supply **128 sorted measured beam elevations in degrees**, one number per line.
   The publisher's `Configuration.json` contains mounting pose/FOV, not a beam table.
   Training-only elevation-bin estimation is accepted
   only if its residual/spacing check passes and is not labeled factory ring IDs.
   Estimation covers the observed angular span rather than allocating rows by
   point density, so dense near-horizon returns do not consume the outer rows.
   The unchanged angular error limit, observed residuals, spacing and accepted
   fraction are logged and recorded under `sensor_meta.beam_fit` in the manifest.
   If inference still fails, check the coordinate frame or supply measured angles;
   retries reuse the completed downloads and extraction receipts.
5. Run **Prepare Pandar128**, then select Train, Generate or the OT experiment.
   **Completed preparation** initially selects the latest available server manifest.
   Choose an older preparation or enter a custom server path when needed; manual
   selections persist across operations. Starting requires a nonempty manifest path.
   Discovery excludes unfinished jobs and missing/corrupt manifests and also works
   after restarting the API or client. Whole trace IDs, source SHA-256, prepared-frame SHA-256 and
   quantized range fingerprints reject overlapping splits. The near-duplicate
   fingerprint is a heuristic, not a comprehensive scene-identity proof.
6. Train. `gpus=-1` (the default) selects every visible GPU on the server;
   `gpus=0` selects CPU and a positive count limits the job. More than one GPU
   launches `torchrun`/DDP, one process per GPU for a standalone training job.
   Experiment jobs are queued on disjoint GPU groups; a diffusion job uses DDP
   within its group. `CUDA_VISIBLE_DEVICES` can restrict the GPUs exposed to the API.
   Validation is also sharded across workers without duplicating validation windows.
   AMP, fused CUDA AdamW, TF32 on supported GPUs, pinned transfers, prefetching and persistent data
   workers reduce stalls. `workers=-1` chooses workers from the CPU affinity and GPU
   count; `workers=0` keeps loading in the training process. CPU automatic mode uses
   no loader subprocesses. Accumulation synchronizes gradients on optimizer updates.
   Automatic optimization is enabled by default. Training measures candidate batches
   up to the configured limit (512 by default), selecting useful throughput with
   15% memory headroom and room for AdamW state. DDP ranks agree on a batch that fits
   every assigned GPU. Probing preserves RNG state and does not update model weights.
   A runtime CUDA OOM restarts only the failed training job with a smaller limit;
   dependency errors and model-allocation failures are not retried. Effective batch
   is selected batch × processes × accumulation. Disable **Automatically optimize
   GPU throughput** to choose a fixed batch. Logs identify each rank,
   device, effective batch and data workers; checkpoints record the resolved resources.

Independent generated samples also run concurrently with CUDA AMP on one model replica per
selected GPU, preserving sample order and individual seeds. The replicas are reused
across calibration and evaluation banks. A shared queue lets faster GPUs take more
work. Compatible requests are batched with independent seeds; observed throughput
and free memory guide batch growth. An OOM halves the batch and retries those same
seeds. Small banks use only as many GPUs as there are samples. Changing batch size
or AMP can change floating-point rounding.

Preparation distributes traces across GPU workers and frames across each worker's
GPUs. Archive sizes and available RAM limit simultaneous trace decoding. Stable
nearest-return selection, point labels and cross-split audits are preserved. OT
scoring and neural probes use their assigned GPU. Independent RL controllers and
calibration methods run concurrently; each policy retains its own feedback order.
Downloads, decompression, file I/O, beam fitting and NumPy/SciPy statistics remain CPU
work. Dependency barriers and insufficient independent work can also leave GPUs
idle. Automatic tuning targets useful throughput; it cannot guarantee constant
100% utilization or protect against arbitrary external changes in free GPU memory.

The batch contract is `range/valid/intensity [B,T,H,W]`, optional-intensity flags,
int64 `timestamp_ns`, sequence IDs and sensor metadata. The published scan export
does not preserve per-scan timestamps: caches explicitly identify timestamps as
**nominal 20 Hz, sequence-relative**, rather than observed hardware timestamps.
Invalid rays have zero storage values with `valid=False`; they are excluded from
range/intensity noise losses. Validity has its own diffusion channel and raydrop
loss. Temporal windows never cross traces.

Diffusion training includes noise prediction, raydrop, optional geometry-anchor
and temporal losses. Training uses partial real-ray anchors and null conditioning.
To generate with an independent synthetic scene, provide a numeric NPZ with
`range` and `valid` arrays `[T,H,W]` matching the checkpoint. Complete-scene
generation introduces no returns outside its mask and clips deviations at known
surfaces to the configured tolerance. This is ray-space conditioning, not scene
reconstruction. The original calibration metrics describe sensor-frame persistence;
the optional OT experiment below adds pose-aware checks and a labeled task probe.

PPO is adapted from `on_policy_RL_Agents.py` and `networks_RL_Agents.py` in the
requested existing project. SAC in that source is disabled; this update adds an
independent working SAC implementation for the new experiment controls.
PPO calibrates bounded range bias/noise, drop probability and intensity scale.
Random search, Bayesian optimization and CMA-ES share the same evaluation budget
and frozen generated samples. Rewards use training traces; five candidate profiles
per method are checked on validation traces. Test arrays are opened for evaluation
after the diffusion checkpoint and calibration profiles have been frozen.

Evaluation includes unmodified training-window resampling, a per-ring empirical
baseline, uncalibrated diffusion and each calibration profile. It reports conditional
range Wasserstein distance, return rates, intensity where available, temporal
statistics, unsupported contexts and bootstrap intervals over complete test traces.
The export profile is chosen on validation. No downstream, commercial-release or
cross-sensor transfer claim follows automatically from these statistics.

Use **Generate** with an own `checkpoint.pt`, optionally its `profiles.json` and a
profile name such as `ppo`. Exports include checkpoint provenance, runtime dependency
lock, notices, PPO policy, calibrated profiles, metrics and numeric sample NPZ files.
Model checkpoints contain the source commit, adapter hashes, training-manifest hash,
sensor configuration and license evidence. Foreign pretrained checkpoints are rejected.

Headless jobs use the same libraries:

```bash
python -m lidarpoc.runner --request configs/prepare.json --output runs/prepare
# Edit configs/train.json to use the returned manifest path.
python -m lidarpoc.runner --request configs/train.json --output runs/train
```

## OT, diffusion and RL experiments

Prepare the data, then choose **OT + diffusion + RL experiment** in the existing
client UI and enter the returned server manifest path. The original Train and
Generate operations remain available. The new workflow runs E0–E5 across the
selected seeds, including conditional debiased Sinkhorn, unequal-mass KL transport,
geometry/temporal constraints and a frozen-generator controller comparison.
SAC, contextual bandit, PPO, fixed controls, random search, BO and CMA-ES are available.

For a headless run, edit the server manifest path and resource limits in
`configs/experiment.json`, then run on the server:

```bash
python -m lidarpoc.runner --request configs/experiment.json --output runs/ot-experiment
```

The example includes seven controllers and three seeds. Development arms,
controller fits, label-budget probes and final evaluations run concurrently on
disjoint GPU groups. Within each phase, available workers take the next ready job.
All development jobs finish and one global selection freezes before any real test
arrays are decoded. Frozen policies preserve their feedback order. Generation-call counts,
denoiser evaluations and measured elapsed allocation are exported; the controller
time cap is checked between actions. A final action can overrun it, and such a run
cannot pass the equal-budget RL gate.

Default scenes are generated locally from road planes, lane markings and boxes,
with separate train/development/test geometry. Alternatively supply three separate
lists of canonical simulator NPZ files. OT reference features come only from real
training windows. All policies and arm selections freeze before real test arrays
are decoded. Test-frame hashes are checked earlier for integrity.

Task transfer evaluation is opt-in. Reprepare older caches to retain native labels;
confirm their semantics before enabling the point-segmentation probe. To enable
real pose metrics, also confirm the odometry convention during Prepare. Missing
task/pose evidence leaves success gates pending. Reduced OT alone is not a task
transfer result. This is a small point-wise probe, not a production detector.

Generate can apply an exported `experiments/seed_<n>/E5_<method>/controller.pt`
with its exact E4 checkpoint and a canonical scene NPZ. The bundle includes its
frozen training reference and rejects another checkpoint or manifest. Existing
PPO calibration profiles remain supported. See [implementation details and artifact
formats](docs/ot_report_implementation.md) for the ablation matrix, fixed units,
optional warm starts, provenance, budgets and remaining limits.

## Libraries and verification

`lidarpoc/data/download.py` handles pinned downloads, selective extraction and
reuse. The other `data/` modules handle NPZ, transforms, projection and splits. `models/` holds the
LidarDM adapter and masking. `methods/` keeps each calibration method independent.
`train.py` handles optimization/DDP; `calibration.py` handles PPO's environment;
`eval/domain_metrics.py` holds metrics. `runner.py` orchestrates these libraries.
`ot/` keeps transport, references, constraints and controller integration separate;
`models/ot_adapter.py` adds losses without changing the UNet capacity. `experiment.py`
orchestrates the new matrix, and `eval/` contains the task, temporal and decision modules.
`logs.py` handles process output, progress and log cursors; `telemetry.py` reads
NVML independently of the training stack. `server.py` manages remote jobs, and
`app.py` is a thin client.

GPU discovery/VRAM configuration and the DataParallel wrapper are adapted from
`PINNs/gpu_optimizer.py` and `PINNs/sandbox_optimizer_lib.py`. The tunnel adapts
`PINNs/ssh_tunnel.sh`, using private control sockets instead of broad process kills.
Source/data notices and revision evidence are in `LICENSES/`. Publisher MIT evidence
is preserved; dependency and collection-rights review remains pending, as in the report.

Install `pytest`, `httpx` and `requirements-client.txt` alongside the server dependencies:

```bash
python -m pytest -q
```

Tests use small synthetic fixtures in the publisher's scan schema. They exercise
projection/masks, split and checkpoint gates, all calibration methods, real random
diffusion training, own-checkpoint reload, geometry conditioning, API jobs and UI
submission, pinned downloads, selective extraction, interrupted extraction recovery,
the port-9000 tunnel configuration, shared-connection ownership, incremental logs,
GPU telemetry/graphs and data preparation isolated from optional model packages.
They also cover differentiable balanced/unbalanced transport with unequal masses,
training-only references, every new controller, frozen-policy generation, E0–E5,
labeled transfer budgets and two-process CPU DDP with OT enabled.
A full Pandar128 training run and CUDA multi-GPU
performance require the remote dataset and hardware. No large dataset or pretrained
artifacts are bundled.
