---
license: mit
pretty_name: Pandar128 Lane Line Detection
task_categories:
  - image-segmentation
tags:
  - lidar
  - autonomous-driving
  - lane-detection
  - lane-line-detection
  - point-cloud
  - 3d
  - adas
size_categories:
  - 10K<n<100K
---

# Pandar128 Lane Line Detection Dataset

**Pandar128** is the largest public dataset for lane line detection using a
**128-beam LiDAR**. It contains synchronized camera frames and LiDAR scans
recorded in real-world driving conditions in Germany, together with **full
sensor calibration** (intrinsics, extrinsics, distortion) and **synchronized
odometry** — enabling projection, sensor fusion, temporal accumulation, and
map-level lane modeling.

This dataset accompanies the paper *"Pandar128 dataset for lane line detection"*
(Beránek, Diviš, Gruber — University of West Bohemia).

- 📄 Paper: *Pandar128 dataset for lane line detection*
- 💻 Code: https://github.com/filipberanek/lane_lines_dataset

## Highlights

- First lane detection dataset captured with a **128-beam LiDAR** (vs. 64-beam in K-Lane / KITTI-360).
- **29 driving traces**, each a ~60 s continuous segment — highway, expressway and urban; sun, cloud and rain; low/mid traffic; construction zones.
- **Two annotation types**: dense point-cloud **semantic segmentation** and lightweight **polyline** annotations (ordered `(x, y, z)`).
- **Odometry + full calibration** included — a first for LiDAR lane datasets.

## Sensor setup

- **Camera**: 2896×1876 px, 30/15 FPS, FOV 120°×73°, mounted behind the windshield.
- **LiDAR**: Hesai Pandar128, roof-mounted, 360° coverage, vertical FOV 40°, 20 FPS, range up to 200 m.
- **GPS/IMU**: 20 Hz odometry (relative transforms between consecutive frames).
- **Region of interest (ROI)**: 0–40 m ahead, ±15 m laterally.

Full calibration is provided in [`Configuration.json`](./Configuration.json).

## Files

The dataset is split into three archives. Download only the part(s) you need.

| File | Content | Approx. size |
| --- | --- | --- |
| `train_part1.zip` … `train_part4.zip` | 17 training traces (split into <50 GB parts) | ~127 GB total |
| `validation.zip` | 6 validation traces | ~39 GB |
| `test.zip` | 6 test traces | ~35 GB |
| `Configuration.json` | sensor calibration (camera + LiDAR) | small |

> The training split is delivered as several parts (`train_part1.zip`,
> `train_part2.zip`, …) because Hugging Face caps a single file at 50 GB.
> Download all of them and extract each — they all share the same `train/`
> folder, so extracting them into one directory reconstructs the full split.

Each archive expands to `&lt;split&gt;/&lt;trace&gt;/…`:

```
<split>/<trace>/
├── <trace>_scan_list.npz                 # raw LiDAR point cloud per frame
├── <trace>_video_list.npz                # raw camera frames
├── <trace>_odom_list.npz                 # 20 Hz odometry (relative transforms)
├── <trace>_polyline_annotation_list.npz  # ground-truth lane polylines
├── <trace>_original_resolution.json      # BEV meshgrid config (resolution, cell size, FOV, padding)
├── <trace>_segmentation_inputs/*.npy     # per-frame BEV meshgrid (LiDAR intensity)
└── <trace>_segmentation_targets/*.npy    # per-frame BEV labels (0=bg, 1=white, 2=yellow)
```

## Annotations

- **Semantic segmentation** — label per cell/point: `0` background, `1` white lane line, `2` yellow lane line.
- **Polylines** — ordered `(x, y, z)` points lying within the physical extent of each lane line; can be rasterized into a BEV meshgrid.
- **Rules** — annotate ego + adjacent lanes in the ego direction, include visible exit/acceleration lanes, skip occluded lines and barrier-separated opposite traffic.

## Splits

Stratified by road type, weather, traffic and construction. The validation and
test sets (6 traces each) each contain at least one challenging condition
(rain, mid-traffic, construction zone, or urban driving).

## Usage

```python
from huggingface_hub import hf_hub_download
import zipfile, numpy as np

# download one split (resumable, cached)
path = hf_hub_download(
    repo_id="filipberanek/pandar128-lane-line-detection",
    filename="test.zip",
    repo_type="dataset",
)

# extract and load a single frame
with zipfile.ZipFile(path) as z:
    z.extractall("pandar128_test")

mesh = np.load("pandar128_test/test/<trace>/<trace>_segmentation_inputs/<trace>_0.npy")
lbl  = np.load("pandar128_test/test/<trace>/<trace>_segmentation_targets/<trace>_0.npy")
polys = np.load("pandar128_test/test/<trace>/<trace>_polyline_annotation_list.npz",
                allow_pickle=True)["data"]
```

> Note: this dataset ships raw `.npy` / `.npz` inside zip archives, so the Hub
> dataset viewer and `datasets.load_dataset()` are not available — download and
> unzip the split(s) you need as shown above.

## Privacy

Recorded on public roads in Germany. **Faces and license plates were anonymized
by the data-collection provider** prior to release, and the authors hold the
corresponding approval for publication.

## Citation

```bibtex
@inproceedings{beranek2025pandar128,
  title     = {Pandar128 dataset for lane line detection},
  author    = {Ber{\'a}nek, Filip and Divi{\v{s}}, V{\'a}clav and Gruber, Ivan},
  year      = {2025}
}
```

## License

Released under the **MIT License** (see [`LICENSE`](./LICENSE)).

## Acknowledgements

Supported by the University of West Bohemia grant **SGS-2025-011** and the
**e-INFRA CZ** project (ID: 90254), funded by the Ministry of Education, Youth
and Sports of the Czech Republic.
