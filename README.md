# Pandar128 LiDAR generation PoC

Implements `clean_license_lidar_poc_report.pdf` and adds the optional experiment
workflow from `sim_to_real_OT_diffusion_RL_updated_report.pdf`, with **RL retained**.
The Streamlit UI runs on the client. The API, dataset reader, diffusion training,
calibration and evaluation run on the remote server through an SSH tunnel.

The model is a **range-image adaptation of LidarDM's conditional UNet frontend**,
trained from random initialization. It does not reproduce the upstream latent
SDF/VAE world-generation pipeline. No KITTI-360/Waymo weights, pretrained VAEs,
external asset banks or LoRA bases are loaded or downloaded.

## Contents

- [Quickstart](#quickstart)
- [Logs, GPU graphs and existing SSH connections](#logs-gpu-graphs-and-existing-ssh-connections)
- [Prepare the dataset](#prepare-the-dataset)
- [Automatic GPU scheduling](#automatic-gpu-scheduling)
- [Training, calibration and generation](#training-calibration-and-generation)
- [RL method library](#rl-method-library)
- [OT, diffusion and RL experiments](#ot-diffusion-and-rl-experiments)
- [Files and API](#files-and-api)
- [PDF results reports](#pdf-results-reports)
- [Troubleshooting](#troubleshooting)
- [Libraries and verification](#libraries-and-verification)

## Quickstart

Use a local copy of this project on both machines. Run each command from that
machine's project directory. Python 3.11+ is required; the SSH helper uses Bash and
OpenSSH on the client. The server needs sufficient RAM and disk for the dataset,
and a compatible NVIDIA driver and CUDA-enabled PyTorch for GPU execution.

| Component | Machine | Address |
| --- | --- | --- |
| Streamlit UI / browser | Client | `http://127.0.0.1:9000` |
| SSH forward to the API | Client | `http://127.0.0.2:9000` |
| FastAPI compute service | Server | `http://127.0.0.1:9000` |

### Server

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

### Client

Create a separate client environment; no GPU or model packages are needed there:

```bash
python3 -m venv .venv-client
source .venv-client/bin/activate
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

Check the API through the tunnel from another client terminal:

```bash
curl --noproxy '*' http://127.0.0.2:9000/health
```

A successful response contains `"status":"ok"`. The API's interactive reference is
available through the tunnel at `http://127.0.0.2:9000/docs`.

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
Use one Uvicorn worker. The API accepts one top-level compute job at a time;
independent tasks inside that job can use multiple GPUs concurrently. Starting a
second job returns HTTP 409. `LIDARPOC_RUNS=/data/lidar-runs` changes artifact
storage when set before starting the server.

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

## Prepare the dataset

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
   files. Each preparation worker decodes one source trace in RAM and writes
   numeric per-frame caches. Available RAM limits concurrent trace workers. Source
   traces may be large even when a small frame limit is selected; `max_frames`
   limits prepared frames per trace, not the archive download size.
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
6. Choose **Train diffusion + calibration** for the original diffusion/PPO workflow,
   **Generate** for an existing application checkpoint, or **OT + diffusion + RL
   experiment** for the ablation and controller matrix. Data, checkpoint, scene
   and output paths refer to the server filesystem. SSH key and control-socket
   paths refer to the client filesystem.

## Automatic GPU scheduling

Automatic scheduling and throughput tuning are enabled by default. These settings
apply to UI submissions and JSON requests:

| Setting | Default | Meaning |
| --- | --- | --- |
| `gpus` | `-1` | All GPUs visible to the server; `0` selects CPU, a positive value limits the count. |
| `auto_optimize` | `true` | Tune training and sampling batches for measured throughput and available VRAM. |
| `batch_size` | `512` | Per-GPU batch ceiling while tuning; fixed requested batch when tuning is disabled. |
| `workers` | `-1` | Choose training data-loader workers automatically; `0` loads inline. |
| `gradient_accumulation` | `1` | Number of microbatches per optimizer update. |
| `rl_enabled` | `true` | Keep calibration methods, including PPO, enabled for the original Train workflow. |

`gpus=-1` selects every visible GPU on the server. More than one GPU
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

## Training, calibration and generation

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

**Generate** fills its fields from the **last successful training run** by default.
It selects that run's frozen manifest, checkpoint, sampling steps, sample count,
seed, throughput optimization settings, and `profiles.json` with the profile
selected on validation. A newer dataset preparation does not replace the training
manifest. The inherited GPU count is capped by the GPUs currently visible on the
server; no visible GPUs selects CPU. An OT controller is filled only when that run
explicitly recorded an existing generation controller. Calibration policy files
such as `ppo_policy.pt` are not substituted for OT controllers. If the run produced
no controller, the optional field remains empty.

Manual edits persist for the current source run, including when switching
operations. **Use detected paths** restores all Generate fields to the last
successful training defaults. A newly completed training run refreshes the default
set together, preventing a mixture of unrelated manifests and saved artifacts.
When selecting another manifest manually, checkpoint discovery still requires a
matching frozen manifest. If no completed report is available for that manifest, it
also checks saved `checkpoint.pt` files in older/interrupted jobs and experiment
subfolders. Recovery requires matching recorded manifest metadata, a recorded
training epoch and a nonempty checkpoint file. It skips active jobs and untrained
baselines; the normal checkpoint provenance checks still run before generation.
With **Automatically extract a real scene
from the prepared dataset** enabled, it also fills the scene field with a real
training window matching that checkpoint's frame count. The server creates or
reuses this small scene cache under
`datasets/conditioning/<manifest_sha256>/window_<T>/real_scene.npz`, using the
first complete window from the **training split**. No model weights are loaded
to populate these fields. The job saves `real_scene.npz` alongside `sample_000.npz`, `sample_001.npz`,
and the remaining synthetic variants of that same window. No manual extraction
script is needed. The real scene contains copied recorded LiDAR data, not newly
generated real-world measurements or RGB images. Its original ranges, masks,
intensities, labels and timestamps are retained where available; beam metadata is
embedded so it can be opened directly in `view_syn_npz.py`.

A supplied scene path overrides automatic extraction and is also exported as
`real_scene.npz`. Manual checkpoint and scene entries are remembered across
operation changes. Click **Use detected paths** to restore automatic choices;
clearing the checkpoint field also restores automatic checkpoint selection.
To retain unconditional generation, disable automatic extraction
and leave the scene path blank (`"auto_conditioning": false` in JSON requests).
Automatic extraction never selects validation/test frames. The result JSON records
the exported scene path and its source (a cached/manual scene file or extracted
training trace/frame records), and the artifact ZIP
contains both the conditioning scene and synthetic outputs. Use the same viewer
and frame index to compare them:

```bash
# Run on the client after extracting the downloaded artifact ZIP:
python view_syn_npz.py real_scene.npz --frame 0 --output real_scene.html
python view_syn_npz.py sample_000.npz --frame 0 --output synthetic_scene.html
```

The viewer opens a nearby scene view (±30 m in X/Y) with the plotting grid
hidden. Use the extent dropdown for **Full scene**, the camera dropdown for top
or axis views, and **Toggle axes / grid** for coordinates. Long-range returns can
otherwise make nearby structures appear tiny. These controls change only the
display; the NPZ and range/intensity images retain their data. To start with all
extents visible, add `--view-radius 0`. Regenerate previously exported HTML files
with the updated viewer to get these controls.

The HTML also embeds matching recorded-camera, real-LiDAR-overlay and synthetic-
LiDAR-overlay images when the downloaded `overlays/` folder is beside the NPZ.
Keep that folder when extracting the artifact ZIP. Frame and sample selection use
`overlays/manifest.json`; the exported image filenames also work without a
manifest. Images are embedded in the HTML, so the finished HTML works offline
without separate image files. For images stored elsewhere:

```bash
python view_syn_npz.py /home/r/lidarpoc/sample_000.npz --output sample.html \
  --overlay-dir /path/to/downloaded/overlays
```

If camera generation failed, the HTML and terminal show the reason from the
downloaded `result.json`. A LiDAR NPZ contains no RGB image by itself. Rebuild
camera overlays on the compute server with the command below, download the
updated artifacts again, then rerun the viewer; no diffusion retraining or
sampling is required.

LiDAR scan rings and gaps between returns remain visible, especially at longer
ranges. The viewer displays measured/generated points rather than RGB images or
textured surfaces. Compare the same frame in `real_scene.npz` and `sample_000.npz`
to distinguish acquisition sparsity from geometry or returns lost by generation.

**Camera overlays for synthetic data.** In **Generate**, leave **Overlay synthetic
LiDAR on recorded camera images** enabled and use real-scene conditioning.
The worker exports the generated LiDAR on the corresponding recorded RGB image,
plus a real-LiDAR overlay for comparison. In **Monitor and results**, select the
overlay frame and synthetic sample. The background is a recorded camera image;
this feature generates LiDAR overlays, not new RGB observations. The artifact ZIP
includes `overlays/camera_*.jpg`, `real_frame_*.png`, `sample_*_frame_*.png`,
`calibration.json` and a timestamp/projection manifest inside `overlays/`.

Camera extraction is deferred until generation, and extracts only the conditioning
trace's `*_video_list.npz` and any missing `*_odom_list.npz` from the pinned cached
archives. Existing scan-only datasets therefore need no new preparation or
training. If an archive was removed, the download receipt identifies the pinned
archive to fetch again. For a manually supplied dataset without cached archives,
place the matching camera and odometry NPZ files beside the source scan. Numeric
camera records are memory-mapped; compressed numeric members use temporary disk
space. Large object-array variants require sufficient RAM and are rejected before
loading if their uncompressed member exceeds 40% of available RAM.

Pairing follows the publisher's [timestamp-based camera selection](https://github.com/filipberanek/lane_lines_dataset/blob/main/article_visualizations/create_videos.py):
use the source LiDAR frame's odometry timestamp and find the closest camera timestamp. Camera
and LiDAR frame indices can differ. The default timestamp unit is seconds and the
maximum difference is 50 ms. Explicit `timestamp_s`, `timestamp_ms`,
`timestamp_us` and `timestamp_ns` fields are also supported, with their units
applied independently to each stream. In the published variant with odometry `t`
and camera `timestamp_us`, leave the configured unit at seconds for odometry;
camera microseconds are converted automatically. The configured timestamp unit
under **Camera overlay settings** applies only to generic `t` fields. The log and
per-frame output metadata record each selected field and its unit. Nominal prepared 20 Hz timestamps are never
used to pair images. New real/synthetic NPZ files retain `source_scan_file` and
`source_frame_indices`; old automatic scene caches can recover this provenance
only when their ranges and masks exactly match the first training window. An
independent/unconditional scene has no corresponding recorded image. Missing
camera data, timestamps or provenance produces a visible explanation in the UI,
logs and report while keeping LiDAR outputs usable.

Projection removes points behind the camera/outside the image and keeps the
nearest return per pixel. Real and synthetic overlays share depth colours and
camera geometry. The bundled mounting interpretation assumes vehicle/body axes
X forward, Y left, Z up, ZYX yaw/pitch/roll, then converts to optical X right,
Y down, Z forward. The default rational distortion uses OpenCV ordering
`[K1,K2,P1,P2,K3,K4,0,0]`, with K4 in the denominator. The publisher's mounting
file does not explicitly establish these conventions; compare the real overlay
against visible scene edges before treating registration as accurate. The
assumptions are recorded in the output. Timestamp matching does not compensate
vehicle/object motion, rolling shutter or all RGB occlusions.

For verified calibration or rectified images, supply an explicit camera JSON in
**Camera overlay settings**. It needs `camera_from_sensor` (rigid 4×4 transform
from the prepared LiDAR sensor frame to optical coordinates), `camera_matrix`
(3×3 intrinsics), `image_size` (`[width,height]`), `model` (`rational` or
`pinhole`) and optional `distortion` (0, 4, 5 or 8 OpenCV-ordered coefficients).
`overlays/calibration.json` uses this portable schema. Intrinsics scale for
uniform image resizing; cropped images need their own calibrated intrinsics.

To add overlays to an existing completed Generate job without rerunning diffusion,
run on the compute machine with the dataset and updated server dependencies:

```bash
python -m lidarpoc.camera_overlay --job runs/GENERATE_JOB_ID
# Override calibration or timestamp units when needed:
python -m lidarpoc.camera_overlay --job runs/GENERATE_JOB_ID \
  --calibration /path/to/verified_camera.json --timestamp-unit s --max-skew-ms 50
```

This writes the images and updates that job's `result.json` and PDF. Refresh the
client page to reload a previously cached result. Install `requirements-server.txt`
after updating; Pillow is the only additional dependency for this feature.

This automation supplies conditioning geometry; it does not establish that the
diffusion model has learned realistic scenes. The existing surface-deviation and
ray-mask constraints still apply.

The RL library adapts the working policy-gradient, deterministic-control and DQN
agents from the requested existing project. SAC in that source is disabled;
Syn retains its independent working SAC implementation. RL methods calibrate
bounded range bias/noise, drop probability and intensity scale.
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
profile name such as `ppo`. **Own checkpoint path on the server** is required and
is filled automatically when a matching completed training job is available.
You can override it with the `checkpoint` value from a completed training job's
`result.json`, or its job directory; the API and headless runner resolve that
directory to `checkpoint.pt` and check that the file exists. Prepare
creates dataset caches and a manifest; creating a scene NPZ supplies conditioning
geometry. Neither operation trains the model. The checkpoint path is remembered
when switching operations in the same client session. The Generate button stays
disabled until both the prepared manifest and checkpoint paths are filled in;
a message beside the button identifies the missing fields. Input lookup failures
identify an outdated/unavailable API, and unsuccessful checkpoint
searches report the actual server runs folder. Automatic real-scene
extraction does not remove the checkpoint requirement. If no trained
checkpoint exists, run **Train diffusion + calibration** with the selected manifest
first. Exports include checkpoint provenance, runtime dependency
lock, notices, fitted RL policies, calibrated profiles, metrics and numeric sample NPZ files.
Model checkpoints contain the source commit, adapter hashes, training-manifest hash,
sensor configuration and license evidence. Foreign pretrained checkpoints are rejected.

Headless jobs use the same libraries:

```bash
python -m lidarpoc.runner --request configs/prepare.json --output runs/prepare
# Edit configs/train.json to use the returned manifest path.
python -m lidarpoc.runner --request configs/train.json --output runs/train
```

## Paired generation: without training and with training

In **Generate**, enable **Compare without training / with training (one output each)**.
Keep the detected manifest/checkpoint from a successful **Train diffusion + calibration**
job and automatic real-scene extraction enabled. The checkpoint must record at least
one completed training epoch. Set the sampling seed, untrained initialization seed
and comparison voxel size, then click **Start remote job**.

The job produces exactly two synthetic NPZ windows:

| File | Diffusion weights |
| --- | --- |
| `sample_000.npz` | Fresh random initialization of the same architecture; no checkpoint weights loaded |
| `sample_001.npz` | Weights from the selected trained checkpoint |

Each file contains the checkpoint's temporal frame count; it is not one file per
frame. Both arms use the same real conditioning, sampling seed, DDIM steps and
geometry constraints. The untrained initialization is reproducible from its own
seed, but is not claimed to reproduce the trained model's historical initial
weights. Calibration profiles and learned OT controllers are bypassed in this mode
to isolate the diffusion weights. Their saved UI fields remain available when you
turn comparison off; existing training, RL and ordinary generation still work.
With at least two selected GPUs, the two arms run concurrently on separate GPUs;
with one GPU or CPU, they run sequentially. Two samples do not provide independent
generation work for more than two GPUs. Analysis and HTML rendering run on the
compute machine; the client receives the finished viewer through the SSH tunnel.

Under **Monitor and results**, pixel-error bars, occupancy bars, pixel maps and
error-distribution graphs are displayed directly. Select **Pixel graph frame**
to change the temporal frame. Click **Open side-by-side images, point clouds and
gap graphs** for the combined viewer. It has synchronized 3D cameras, nearby/full-scene extents,
and a temporal-frame selector. Download the artifact ZIP to open the self-contained
`comparison.html` offline, without access to the dataset or an internet connection.
The two image panels are LiDAR overlays on the **same recorded camera photo**;
the application does not generate new RGB photographs. The real camera and real
LiDAR overlay are available as references in the viewer.

All three pairs are reported: untrained versus real, trained versus real, and
trained versus untrained. The graphs and machine-readable scores include:

- Camera-pixel optical-depth MAE/RMSE/p95, return occupancy IoU and disagreement.
- LiDAR beam/azimuth-grid radial-range errors and return-mask differences, available
  even when timestamp-matched camera data is missing. These are not RGB-camera pixels.
- Per-pixel overlay RGB MAE maps and foreground RGB errors/histogram discrepancies.
  The common photographic background and label strip are excluded from foreground
  scoring. These values measure overlay differences, not synthetic RGB quality.
- Voxel occupancy IoU, return-density differences, Jensen-Shannon distance and total
  variation on a shared sensor-frame grid with origin `(0,0,0)`.
- Symmetric mean nearest-neighbour distance in unsquared metres, plus range, height
  and available intensity distributions using identical bins.

Lower errors/distances are better; higher IoU is better. Missing support produces
`null`/n/a, including depth errors with no jointly observed pixels. Maps may be
reduced for display and point plots subsampled, but metrics use every valid return
and full-resolution camera pixels. Jensen-Shannon is the **square root** of the
divergence, using log base 2 (0–1), following the
[SciPy definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html).
Nearest-neighbour distances use exact
[cKDTree queries](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html).
These are descriptive discrepancies for a paired scene, not a population-level
covariate-shift test. Conditioning masks and range clamping constrain geometry in
both arms, so a low geometry error alone does not prove successful learning.

The job stores `comparison.json`, full-resolution `comparison/maps_*.npz`,
`comparison.html`, `real_scene.npz`, both synthetic NPZs, and matching
`overlays/sample_000_frame_*.png` / `sample_001_frame_*.png`. The results PDF includes
the paired metrics, pixel PNG graphs and their limits. Install the updated `requirements-server.txt`
for server-side Plotly rendering. Camera provenance, timestamps and calibration
requirements are the same as for ordinary Generate. If camera matching fails,
point-cloud/voxel results, LiDAR range-grid pixel graphs and the viewer are retained.
Missing camera images and camera-pixel scores are explicitly reported as unavailable.

**Pixel graphs for every conditioned Generate job.** Both ordinary Generate and
untrained/trained comparison export `pixel_graphs.html`, `pixel_graphs.json`,
`pixel_graphs/metrics.csv`, full-resolution `pixel_graphs/maps_*.npz`, and portable
PNG map/distribution graphs for every frame. No browser or additional image-export
dependency is needed to create the PNGs. The artifact ZIP includes all these files.
Use **Prepare pixel graphs download**, then **Download pixel PNGs, HTML, CSV and maps**
for a smaller ZIP without model checkpoints. Open `pixel_graphs.html` offline for
interactive plots, or open the PNGs directly. The single-NPZ viewer embeds these
PNG graphs when `pixel_graphs.json` and `pixel_graphs/` are beside the NPZ files.

For an existing run, update/restart the client and API and click **Build / refresh
pixel and OT graphs**. This reads the saved NPZ and camera overlay files; it does not train
or sample a model. The UI refreshes its cached JSON, PNG, PDF and ZIP data afterward.
Artifact downloads also backfill missing graphs for older completed Generate jobs.
To build the same outputs from an extracted download, run:

```bash
python -m lidarpoc.pixel_graphs --job /path/to/extracted/job
```

Ordinary generation compares each saved sample against `real_scene.npz`; it never
labels two ordinary samples as untrained/trained. Camera maps require the saved
`overlays/calibration.json`, camera records and matching overlay PNGs. If these are
unavailable, range-grid graphs remain visible and the camera diagnostic is shown.
Unconditional jobs without a real reference report why pixel gaps cannot be computed.

**Optimal-transport distribution graphs.** Install the updated server requirements
(or `python -m pip install 'POT>=0.9.6,<0.10'` in the environment used for offline
analysis). Reports now include the following distribution measures:

| Measure | What is compared / reported |
| --- | --- |
| Wasserstein-1 / Wasserstein-2 | Exact empirical 1D distances for all finite radial ranges, intensities when present, camera depths and overlay luminance; linear-program OT on sampled multivariate supports |
| Sliced Wasserstein-2 | Monte Carlo distance over 64 reproducible random projections of each multivariate support |
| Debiased Sinkhorn | Squared-cost KL-regularized cross-objective minus half each self-objective; distinct from the raw entropic transport cost |
| Gromov-Wasserstein | Square root of the squared internal-distance distortion objective, with no factor of 1/2; a best-found nonconvex solution, not a certified global optimum |
| Fused Gromov-Wasserstein | Normalized objective combining internal-distance structure and feature mismatch, default alpha 0.5 |
| Unbalanced KL-OT | Full regularized objective with relaxed masses; cloud masses retain return probability per fixed ray exposure |

Cloud W1/W2, sliced-W2 and GW square-root loss are in **metres**; cloud Sinkhorn
divergence is in **m²**. The solver uses one shared sensor maximum-range scale for
both clouds, then converts those values back. FGW/UOT remain normalized objectives.
These objectives are not all mathematical metrics, and regularized UOT need not be
zero for identical inputs. The graphs label units per measure; compare arms within
the same space and measure rather than comparing unlike units.

Image OT uses normalized `[u,v,R,G,B]` features on one common projected foreground
for every arm, excluding label rows. GW therefore describes the internal relations
of those spatial/appearance features. FGW adds RGB mismatch. Both photographs share
the recorded background, so these scores assess overlays, not newly synthesized
photographs. Camera scores are unavailable when calibrated image pairing is missing;
cloud/range-distribution graphs still work. Wasserstein compares value distributions
and can detect different changes from the separate aligned pixel-error maps.

Multivariate OT uses a uniform seeded subset of **128 points per distribution** by
default, with weights normalized to a probability distribution for balanced OT.
LP distances are exact on that bounded support only, not on the complete point cloud.
Sampling error can be material; increase the support limit when needed. GW/FGW use
product and coordinate-W2 plan initializations and retain the lower feasible objective.
Solver warnings, marginal residuals, iteration/stopping diagnostics, selected
initialization, parameters and sampled indices are saved in the JSON/NPZ outputs.
Sinkhorn retries difficult solves with warm-started potentials up to ten times the
base iteration limit; marginal residuals remain checked. UOT costs explicitly use
generalized KL with the mass correction terms, rather than a normalized-KL shortcut.
Failure or missing support produces n/a with its reason; finite local GW results
are labelled approximate. Small negative floating-point roundoff is clipped only
within the documented numerical tolerance; significant invalid objectives are null.

The UI, pixel HTML viewer, combined comparison viewer, single-NPZ viewer and PDF
include OT bar graphs and empirical scalar CDF plots. The CSV has `ot/<space>` rows
with a unit column. The frame NPZ additionally stores sampled supports, normalized
cost matrices and W1/W2/GW/FGW/UOT couplings. PNGs are included in both download ZIPs.
Rebuild existing client-side results without training or a server connection:

```bash
python -m pip install 'POT>=0.9.6,<0.10'
python -m lidarpoc.pixel_graphs --job /path/to/extracted/job --ot-points 128 --ot-projections 64
xdg-open /path/to/extracted/job/pixel_graphs.html
```

Support and projection counts accept 8–512. Other solver settings and the seed are
retained in `pixel_graphs.json`; the initial seed comes from the saved job request.
Use an isolated environment with consistent NumPy-compatible dependencies as in
the quickstart. Evaluation uses NumPy/POT CPU solvers; it does not change the existing
differentiable OT training losses or RL methods. Metric definitions follow
[POT's linear OT API](https://pythonot.github.io/gen_modules/ot.lp.html),
[Gromov-Wasserstein API](https://pythonot.github.io/gen_modules/ot.gromov.html),
[unbalanced OT API](https://pythonot.github.io/gen_modules/ot.unbalanced.html) and
[debiased Sinkhorn formulation](https://arxiv.org/abs/1810.08278).

For API/CLI requests, keep `action: "generate"`, supply the manifest and trained
checkpoint, and add `"compare_training": true`, `"auto_conditioning": true`,
`"camera_overlay": true`, `"comparison_init_seed": 0`,
`"comparison_voxel_m": 0.25`. Leave `profile_file` and `controller_file` empty;
the sample count is normalized to two. Use the existing runner or `POST /jobs`.

To recompute maps/viewer/PDF for an existing **paired** job without rerunning
diffusion, run on the compute machine (or an extracted artifact folder):

```bash
python view_comparison.py runs/PAIRED_GENERATE_JOB_ID --voxel-size 0.25
```

The command requires the original paired job's `result.json`, both annotated NPZs,
`real_scene.npz` and overlay files. It does not reinterpret two ordinary generated
samples as an untrained/trained experiment. Regenerating camera overlays with
`python -m lidarpoc.camera_overlay --job runs/PAIRED_GENERATE_JOB_ID` also refreshes
the comparison outputs. After regeneration, refresh the client to reload cached
results.

## RL method library

All RL implementations live in [`lidarpoc/rl/`](lidarpoc/rl/), with an independent
`optimize()` entry point per method. Shared modules provide replay, n-step returns,
bounded actions, training loops and frozen-policy loading. The UI and processing
workflows use [`methods/registry.py`](lidarpoc/methods/registry.py); its method names
and configuration types can be imported without loading Torch. Existing imports
from `lidarpoc.methods.ppo` and `lidarpoc.methods.sac` remain supported.

New method modules expose `optimize(objective, seed=0, device='cpu', config=None)`
and return a policy bundle containing CPU tensors. The objective provides `dim`,
`initial`, `remaining`, `history` and `observation(unit)` or `context(unit)`;
`objective(unit)` records and returns a loss. An optional `last_reward` supplies
shaped feedback. LiDAR-specific rewards and data processing stay in
`calibration.py` and `ot/environment.py`.

Every method below is selectable under **Calibration methods** and
**Frozen-generator controllers**:

| Method ID | Action / learning approach |
| --- | --- |
| `ppo` | Existing clipped policy optimization with discrete parameter adjustments |
| `sac` | Existing squashed-Gaussian actor, twin critics and entropy-temperature learning |
| `a2c` | Discrete advantage actor-critic with GAE and bootstrapped rollouts |
| `reinforce` | Discrete Monte Carlo policy gradient |
| `n_step_actor_critic` | Synchronous n-step actor-critic, adapted from the source's A3C-labelled class |
| `ddpg` | Continuous deterministic actor with one critic and target networks |
| `td3` | Continuous deterministic actor, twin critics, target smoothing and delayed actor updates |
| `dqn` | Discrete Q-learning with uniform replay |
| `enhanced_dqn` | Double DQN, dueling network and prioritized replay |
| `n_step_dqn` | Double DQN with n-step return targets |
| `c51` | Categorical distributional Q-learning with 51 return atoms |
| `rainbow` | Source-style Double DQN + dueling + prioritized replay + n-step + C51; no NoisyNet component |

Discrete methods choose one normalized parameter to increase/decrease by 0.1 or
leave unchanged. DDPG, TD3 and SAC make bounded continuous adjustments of up to
0.2. Calibration converts normalized controls to physical parameter bounds. E5
uses guidance, dropout, scenario and sample-count controls; its scenario/sample
coordinates are rounded to discrete choices. Evaluation keeps a fixed common
scenario mixture and one sample per call. Policies are frozen before test use.

The new replay methods learn once at least two replay transitions are available;
their batch is `min(rl.batch_size, replay size)`. Random-action warmup is capped by
half the initially available call budget. N-step tails flush at episode/budget
boundaries. A2C/n-step actor-critic update at rollout boundaries, and REINFORCE at
the end. Logs and exported policies record transition and learning-update counts.
These adaptations allow learning at the default 30-call budget; a small budget
does not establish convergence or improved fidelity.

In the UI, open **Additional RL method settings**. In JSON requests, use:

```json
{
  "methods": ["ppo", "a2c", "td3", "enhanced_dqn", "bayesian"],
  "rl": {
    "warmup": 4,
    "batch_size": 16,
    "replay_size": 10000,
    "rollout_steps": 5,
    "n_steps": 3,
    "value_min": -20.0,
    "value_max": 20.0
  }
}
```

This is a fragment to merge into a Train/Experiment request, not a complete job.
For Experiment, select algorithms in `experiment.controllers`; top-level `methods`
selects Train calibration methods. `rl` settings apply to the ten newly adapted
methods. PPO retains its existing optimizer settings; SAC retains
`experiment.sac_warmup` / `experiment.sac_batch_size` in E5 and its existing defaults
in calibration. Set C51/Rainbow return support for the tested reward scale.

New calibration policies are saved as `<method>_policy.pt`; E5 bundles are saved
under `experiments/seed_<n>/E5_<method>/controller.pt`. Generate supports all frozen
controller formats. Each selected RL controller gets a paired session comparison
against the development-selected fixed/random/BO/CMA comparator, when available,
under `gates.rl_comparisons`; the existing SAC field remains available. PDFs include
the comparisons. No method is automatically removed based on a result.

Independent methods/seeds use the existing GPU task scheduler, while each policy
keeps its own ordered feedback. Source SAC/model-based stubs remain excluded.
The source MultiAgentDQN wrapper duplicates independent agents and is covered by
this scheduling approach. The synchronous n-step method is not asynchronous A3C.

## OT, diffusion and RL experiments

Prepare the data, then choose **OT + diffusion + RL experiment** in the existing
client UI and enter the returned server manifest path. The original Train and
Generate operations remain available. The new workflow runs E0–E5 across the
selected seeds, including conditional debiased Sinkhorn, unequal-mass KL transport,
geometry/temporal constraints and a frozen-generator controller comparison.
All methods in the RL library, contextual bandit, fixed controls, random search,
BO and CMA-ES are available. Existing default selections remain unchanged.

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

## Files and API

Prepared data and run outputs stay on the server. The default layout is:

```text
Syn/
├── app.py                    # Streamlit client
├── server.py                 # Remote FastAPI service
├── ssh_tunnel.sh             # Client-side forwarding helper
├── view_comparison.py        # Rebuild a paired job's metrics and offline viewer
├── configs/                  # Example prepare, train and experiment requests
├── lidarpoc/                 # Independent processing and method modules
├── LICENSES/                 # Captured dataset/source notices and revision evidence
├── datasets/pandar128/       # Downloaded and extracted source data
│   ├── .downloads/           # Reusable compressed archives
│   ├── train/
│   ├── validation/
│   └── test/
└── runs/<job_id>/            # One directory per API job
```

Each API job has `request.json`, `state.json` and `worker.log`; running jobs expose
`progress.json` when available. A successful job writes `result.json`. Additional
files depend on the operation:

| Operation | Main outputs relative to the job directory |
| --- | --- |
| Prepare | `prepared/manifest.json` and per-frame numeric caches |
| Train | `checkpoint.pt`, `training.json`, `profiles.json`, `sample_000.npz`, resource records and any fitted RL policies |
| Generate | `real_scene.npz` when conditioned, `sample_*.npz` and provenance in `result.json` |
| Generate with training comparison | Two annotated synthetic NPZs, camera overlays, `comparison.html`, `comparison.json`, full-resolution `comparison/maps_*.npz` |
| Experiment | `experiments/`, `scenes/`, selection records and per-arm checkpoints/controllers |

Successful Prepare, Train, Generate and Experiment jobs also save `result.pdf`
alongside `result.json`. The UI downloads completed job artifacts as a ZIP,
including both report formats. Raw dataset archives, prepared
caches and `worker.log` stay on the server; read logs through the UI or log API.
Headless jobs write to the `--output` directory instead of an API-generated ID.

All API paths below are relative to `http://127.0.0.2:9000` on the client:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | API status, server hardware and default dataset directory |
| GET | `/metrics/gpu` | Current server GPU telemetry |
| GET | `/dataset/prepared` | Discover completed prepared manifests |
| POST | `/dataset/inspect` | Inspect NPZ schemas; body: `{"root":"/server/path"}` |
| POST | `/generation/inputs` | Return Generate defaults and a real-scene path; `{}` selects the last successful training run, or supply `manifest` to discover matching checkpoints. Optional `checkpoint` and `auto_conditioning`. |
| POST | `/jobs` | Submit a `RunConfig` request; returns the job ID |
| GET | `/jobs` | List recent jobs |
| GET | `/jobs/{job_id}` | Status, progress, training metrics and recent log text |
| GET | `/jobs/{job_id}/logs?offset=0` | Incremental processing logs |
| POST | `/jobs/{job_id}/cancel` | Stop a running job and its worker process group |
| GET | `/jobs/{job_id}/result` | Completed result JSON |
| GET | `/jobs/{job_id}/report` | Completed PDF report; also generates one for older JSON-only results |
| GET | `/jobs/{job_id}/comparison` | Offline HTML viewer for a completed untrained/trained comparison |
| POST | `/jobs/{job_id}/pixel-graphs/rebuild` | Rebuild pixel graphs/CSV/maps/PDF from a completed Generate job and return refreshed results |
| GET | `/jobs/{job_id}/pixel-graphs` | Interactive pixel viewer; backfill missing graphs from saved data |
| GET | `/jobs/{job_id}/pixel-graphs/{filename}` | A generated PNG pixel graph |
| GET | `/jobs/{job_id}/pixel-graphs.zip` | Pixel PNGs, offline HTML, metrics CSV/JSON and full-resolution maps |
| GET | `/jobs/{job_id}/artifacts` | Completed artifact ZIP |

## PDF results reports

In **Monitor and results**, choose a completed job, click **Prepare PDF report**,
then **Download PDF report**. **Download JSON report** remains available. PDFs are
generated on the server and transferred through the existing SSH tunnel; the
client needs no PDF library. Install the updated `requirements-server.txt` before
running the updated server; it includes ReportLab.

The PDF summarizes the selected arm/profile, held-out scores and exported 95%
session confidence intervals, with a comparison chart. Experiment reports include
OT components, return statistics, evidence gates, RL/comparator findings,
allocated compute, batch-size differences and limitations. Prepare reports show
trace inventories; Train reports include loss history; Generate reports show
sample counts and geometry results. Full detailed records remain in JSON.

Convert an existing downloaded JSON without retraining or contacting the server:

```bash
# From this project directory, in a local environment:
python -m pip install 'reportlab>=4,<5'
python -m lidarpoc.report /path/to/result.json --output /path/to/result.pdf
```

Omitting `--output` writes a PDF beside the input JSON. Export preserves the source
JSON, escapes result text, records its SHA-256 and publishes the PDF atomically.
An existing completed server job gets a PDF on its first report/artifact request.

## Troubleshooting

| Symptom | Check / resolution |
| --- | --- |
| `Permission denied (publickey)` | Use the same user, host and `-i /absolute/path/to/key.pem` that work in your normal SSH command. Pass that key to the helper with `-k`, or provide the live control socket with `-S`. |
| Identity file is inaccessible | Verify the path and ownership on the client. On a Unix client, the key must be readable by your account and private, for example `chmod 600 /path/to/your-key.pem`. |
| `mux_client_forward: forwarding request failed` | Check `bash ssh_tunnel.sh status` and the health URL first; an existing forward may already work. Use `ss -ltnp 'sport = :9000'` to identify a conflicting client listener. The UI must bind `127.0.0.1`; the API forward must bind `127.0.0.2`. Remove only the conflicting forward before retrying. |
| Explicit SSH control socket is unavailable | The `-S` path must belong to a running master. Start the reusable connection shown above, or omit `-S` to let the helper establish a managed connection. |
| HTTP 422: missing manifest | Complete Prepare, then select its manifest from **Completed preparation**. JSON requests need the full server path to `prepared/manifest.json`. |
| HTTP 422: generation requires a checkpoint | Enter the `checkpoint` path from the completed training job's `result.json`, using the same manifest. If training has not run, complete **Train diffusion + calibration** first. A conditioning NPZ is a scene input, not a model checkpoint. |
| `IsADirectoryError` when loading a checkpoint | The path must name `checkpoint.pt`, for example `runs/<training_job>/checkpoint.pt`. Updated versions also accept the training job directory and resolve it automatically. Click **Use detected paths** to fill matching checkpoint and scene files. |
| Beam inference fails its residual/spacing check | Confirm `vehicle` versus `sensor` coordinates. Supply 128 sorted measured beam elevations in degrees if inference still fails. Mounting/FOV configuration is not a beam calibration table. |
| GPU utilization is zero during preparation | Downloading, decompression, trace decoding, beam estimation and file writes use CPU/I/O. GPU projection runs between those stages; check the processing log for the current stage. |
| Training uses only one GPU | Keep **Remote GPUs (-1 = all, 0 = CPU)** at `-1` and automatic optimization enabled; check server-side `torch.cuda.device_count()`, `CUDA_VISIBLE_DEVICES` and rank/resource logs. A small sample bank or one ready task may not occupy every device. |
| `HybridCache`, `resolve_revision`, bitsandbytes CUDA binary or FlashAttention undefined-symbol import errors | Use the isolated server environment above with a GPU-compatible PyTorch build and `requirements-server.txt`. This project does not require PEFT, Transformers, bitsandbytes or FlashAttention; incompatible optional packages in a shared environment can break imports. |
| Hugging Face unauthenticated-request warning | Public downloads can continue. Set `HF_TOKEN` in the server environment before launch if you want authenticated Hub requests. |
| HTTP 409 on submission | One API compute job is already running. Wait for completion or cancel it in the UI. |

The GPU graphs report host-wide physical usage. They can include unrelated
processes, and unavailable telemetry fields are distinct from zero utilization.
Inspect `worker.log` and `resources.json` to confirm this job's assigned devices.

## Libraries and verification

`lidarpoc/data/download.py` handles pinned downloads, selective extraction and
reuse. The other `data/` modules handle NPZ, transforms, projection and splits. `models/` holds the
LidarDM adapter and masking. `rl/` contains the RL methods and shared learning
primitives; `methods/` contains the optimizer registry, black-box optimizers and
PPO/SAC compatibility imports.
`train.py` handles optimization/DDP; `tuning.py` profiles batch throughput;
`scheduling.py` assigns independent tasks to GPU groups; `sampling.py` reuses
per-GPU replicas and batches generation requests. `data/prepare_tasks.py` and
`data/gpu_projection.py` distribute trace preparation and projection.
`calibration.py` handles PPO's environment;
`eval/domain_metrics.py` holds metrics. `runner.py` orchestrates these libraries.
`ot/` keeps transport, references, constraints and controller integration separate;
`models/ot_adapter.py` adds losses without changing the UNet capacity. `experiment.py`
orchestrates the new matrix, `experiment_tasks.py` defines its schedulable work,
and `eval/` contains the task, temporal and decision modules.
`logs.py` handles process output, progress and log cursors; `telemetry.py` reads
NVML independently of the training stack. `server.py` manages remote jobs, and
`app.py` is a thin client. `report.py` renders JSON results to PDF independently
of Torch, the model stack and SSH.
`comparison.py` coordinates paired generation and analysis; `eval/spatial_gap.py`
provides independent pixel, voxel and covariate metrics. `comparison_viewer.py`
renders the offline viewer without model imports. `view_comparison.py` rebuilds
the analysis from an existing paired job; it never trains or samples a model.
`pixel_graphs.py` shares pixel analysis and graph rendering between regular and paired
generation, the UI, offline viewers and downloads. It can backfill old saved runs
without loading model weights; PNG rendering uses the existing Pillow dependency.
`eval/transport_gap.py` owns scalar/multivariate OT estimates and solver diagnostics;
`transport_graphs.py` prepares their distribution inputs and shared PNG/interactive
plots. Training/experiment OT modules remain separate from this reporting library.

GPU discovery/VRAM configuration and the DataParallel wrapper are adapted from
`PINNs/gpu_optimizer.py` and `PINNs/sandbox_optimizer_lib.py`. The tunnel adapts
`PINNs/ssh_tunnel.sh`, using private control sockets instead of broad process kills.
Source/data notices and revision evidence are in `LICENSES/`. Publisher MIT evidence
is preserved; dependency and collection-rights review remains pending, as in the report.

Run verification in a local test environment with the server dependencies:

```bash
python -m pip install -r requirements-client.txt pytest
python -m pytest -q
```

The default suite uses synthetic data and supports a CPU-only PyTorch install;
it does not download the full Pandar128 dataset.

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
