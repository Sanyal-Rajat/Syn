# Pandar128 LiDAR generation PoC

Implements `clean_license_lidar_poc_report.pdf`, with **RL retained at your request**.
The Streamlit UI runs on the client. The API, dataset reader, diffusion training,
calibration and evaluation run on the remote server through an SSH tunnel.

The model is a **range-image adaptation of LidarDM's conditional UNet frontend**,
trained from random initialization. It does not reproduce the upstream latent
SDF/VAE world-generation pipeline. No KITTI-360/Waymo weights, pretrained VAEs,
external asset banks or LoRA bases are loaded or downloaded.

## Run

Copy this directory to the remote server. Install a PyTorch build appropriate for
its GPUs, then install `requirements-server.txt` in a Python 3.11+ environment:

```bash
python -m pip install -r requirements-server.txt
python server.py
# Equivalent: uvicorn server:app --host 127.0.0.1 --port 9000 --workers 1
```

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

## Data and experiments

1. Choose **Prepare Pandar128** and click **Start remote job**. With the dataset
   directory left blank, the server automatically downloads the selected
   [Pandar128](https://huggingface.co/datasets/filipberanek/pandar128-lane-line-detection)
   revision `3366842fa701e4e5ec313f9124acb84b502f11ee` into the server project's
   `datasets/pandar128/` folder. Six archives (about 212 GB compressed) are retained
   in `.downloads/`; extraction writes only `*_scan_list.npz` under
   `train/<trace>/`, `validation/<trace>/`, `test/<trace>/`. Allow additional disk
   space for the extracted scans and prepared caches. Downloading and extraction
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
5. Run **Prepare Pandar128**, then copy the returned remote manifest path into the
   training form. Whole trace IDs, source SHA-256, prepared-frame SHA-256 and
   quantized range fingerprints reject overlapping splits. The near-duplicate
   fingerprint is a heuristic, not a comprehensive scene-identity proof.
6. Train. `gpus=0` selects CPU; `gpus>1` launches actual `torchrun`/DDP, one process
   per GPU. AMP, accumulation, validation checkpoint selection and the adapted
   PINNs VRAM heuristic are included. Batch size remains a hardware-dependent
   setting; the heuristic is not an OOM guarantee.

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
reconstruction. Pose-aware temporal warping and downstream perception training
are not implemented; temporal metrics describe sensor-frame persistence.

PPO is adapted from `on_policy_RL_Agents.py` and `networks_RL_Agents.py` in the
requested existing project. SAC there is disabled and is not exposed as working.
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

## Libraries and verification

`lidarpoc/data/download.py` handles pinned downloads, selective extraction and
reuse. The other `data/` modules handle NPZ, transforms, projection and splits. `models/` holds the
LidarDM adapter and masking. `methods/` keeps each calibration method independent.
`train.py` handles optimization/DDP; `calibration.py` handles PPO's environment;
`eval/domain_metrics.py` holds metrics. `runner.py` orchestrates these libraries.
`server.py` only manages remote jobs, and `app.py` is a thin client.

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
and the port-9000 tunnel configuration. A full Pandar128 training run and multi-GPU
performance require the remote dataset and hardware. No large dataset or pretrained
artifacts are bundled.
