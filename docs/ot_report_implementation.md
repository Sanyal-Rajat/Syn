# Updated report implementation

Specification: `sim_to_real_OT_diffusion_RL_updated_report.pdf`, SHA-256
`3cb5b0d8d6cae8d83264a73684a86753b7b1c75cd1ab58523f61591a28ab9ffa`.
The existing application is extended through `action: "experiment"`; Prepare,
Train, Generate, PPO calibration, logs, telemetry and SSH reuse keep their original
entry points. No additional runtime package is required.

Ordinary Train enables per-frame range W1, conditional and unbalanced transport
losses by default (weight 1 each). A bounded training-only reference is prepared automatically.
Generate applies bounded output refinement by default, on the existing sampling
GPUs, and records initial/final hard-export range scores and secondary objectives. Calibration/controller
outputs receive a final refinement pass before export. Existing checkpoints support
generation refinement; new training is required to change their learned objective.
The explicit E0/E1 experiment training controls remain available. Output refinement
uses the same setting across all experiment/comparison arms. See the README's
default distribution-matching section for settings, provenance and output limits.

`ot/range_distance.py` integrates the one-dimensional empirical CDF difference
over all finite positive ranges. Soft return masses provide training gradients;
hard masks yield the same frame W1 used by the simple viewer. Every epoch samples
the same bounded validation windows and seeds using the normal DDIM/refinement
pipeline. The lowest mean supported-frame W1 selects the trained checkpoint;
missing generated frames cannot win. Validation samples are inputs for scoring,
never part of the training reference or weight updates. Scene-conditioned
refinement cannot worsen any supported frame W1 or reduce initial frame return
counts. Secondary OT breaks ties; its score can increase when range W1 improves.
Unconditioned range targets use bounded quantiles from training data only.

## Report mapping

| Requirement | Implementation |
| --- | --- |
| E0, unadapted baseline | Random-root model, or the same approved own checkpoint used by every arm; no new model training; shared generation refinement still applies |
| E1, real-domain diffusion | Same UNet capacity, seed, epochs and approved training traces |
| E2, conditional transport | E1 plus per-frame range W1 and debiased Sinkhorn on fixed physical features |
| E3, return masses | E2 plus KL-regularized unbalanced transport |
| E4, physical/temporal constraints | E3 plus geometry, pose-aware temporal loss when supported, and regularization |
| E5, frozen controls | Frozen E4, metered SAC; fixed/random/BO/CMA-ES/bandit/PPO comparisons |
| Canonical geometry | Procedural first-return road/box scenes or independent simulator NPZ inputs |
| Transfer, uncertainty and gates | Optional semantic probe, identical label budgets, whole-session bootstrap, explicit pending/rejection outcomes |
| Reproducibility | Request, report hash, manifest/reference hashes, library hashes, model/runtime/license evidence and frozen selection |

This remains the project's range-image adaptation of the LidarDM conditional UNet.
The report's suggested SDF/VAE and LoRA modules are not reproduced. Training uses
the existing same-capacity network, starting randomly by default. An experiment's
optional `checkpoint` can be an own checkpoint with valid random-root provenance
and the exact frozen manifest. Architecture settings must match it. All arms start
from that same state; E0 performs no new training. No outside pretrained weights
are accepted. The legacy Train operation still starts randomly by default.

## Transport and physical constraints

`ot/sinkhorn.py` implements differentiable log-domain scaling in Torch. Balanced
conditional transport subtracts both self costs. Unbalanced transport minimizes
the raw transport cost plus entropy and source/target marginal KL penalties;
it does not subtract self costs. Tests check an analytical unequal-mass solution,
zero-mass behavior and autograd finite differences.

The five features are range in metres, normalized intensity, elevation in radians,
local return density and sensor-frame temporal persistence. Default scales are
`[120, 1, pi/2, 1, 1]`; costs are weighted squared scaled distances. Scales and
weights are explicit physical configuration, never fitted on development/test
data. The frozen reference records the feature specification, training trace IDs
and prepared-manifest hash. Automatic reference preparation disables the intensity
weight when observed measurements are missing. The lower-level reference builder
still rejects a positive intensity weight with missing measurements.

Conditional bins use fixed range edges and ring groups. They approximate scene
conditions; they do not establish matched materials, weather or incidence angles.
Training bins use detached target geometry. Sparse/missing bins are reported and
excluded from conditional averaging; absence of any supported training bin fails
the job. Evaluation exposes unsupported support and penalizes it in the selection
surrogate. Point sampling is deterministic and capped per bin.

Mass bins use ring group × azimuth sector × range. Return counts are divided by
fixed emitted-ray exposure, preserving the return probability as their total mass.
They are never renormalized to observed count. Fixed exposure scaling permits
comparison across windows without erasing missing returns. Soft validity makes
this term differentiable during diffusion training. Entropy and marginal relaxation
parameters are explicit (`epsilon`, `tau_source`, `tau_target`).

Scene generation constrains output to the supplied first-return mask and bounds
range displacement by `geometry_tolerance_m`. Dropout may remove returns. Labels
and poses stay associated with their canonical rays. Physical checks cover new
returns, displacement, bounds and beam/ring projection. These ray-space checks
do not certify complete volumetric geometry or dynamic-object simulation.

Temporal checks warp depths using `world_from_sensor` poses, nearest-beam projection
and first-return occlusion masks. Generated residual correlation is measured against
canonical scenes. Confirmed real poses also enable real ego-motion persistence
and alignment residuals. The OT persistence feature itself remains sensor-frame;
pose compensation is a separate constraint/metric. Missing pose evidence is reported
as unsupported rather than replaced with identity transforms.

## Scenes, preparation and task labels

Without external files, the runner ray casts road planes, lane markings and boxes
with separate deterministic seed families for training, development and test scenes.
No external assets are downloaded. External files require separate nonempty
`experiment.scene_files`, `development_scene_files` and `test_scene_files` lists.
Repeated geometry across scene splits is rejected by hash; this is an exact check,
not a comprehensive scene-similarity audit.

Canonical numeric NPZ schema, matching the checkpoint sensor and window:

| Field | Shape and meaning |
| --- | --- |
| `range` | `[T,H,W]`, metres, finite, visible values within sensor bounds |
| `valid` | `[T,H,W]`, first-return visibility mask |
| `intensity` | Optional `[T,H,W]`, normalized; defaults to zeros |
| `has_intensity` | Optional `[T]` observed-intensity flags |
| `labels` | Optional integer `[T,H,W]`, negative values ignored |
| `world_from_sensor` | Optional rigid `[T,4,4]` homogeneous poses |
| `timestamp_ns` | Optional `[T]` sequence timestamps |

Preparation now stores the native `Classification` field, projected source-point
indices and optional odometry. Old caches still work for old workflows; rerun
Prepare into a new output directory when labels or poses are needed. Downloads
selectively extract available odometry alongside scans, preserving previous ZIP
cache/receipt behavior. Older extraction receipts are upgraded from cached archives.
No camera files are extracted.

Odometry defaults to `disabled`. Explicit choices are `world_from_vehicle` or
`previous_from_current`; the latter accumulates transforms in sequence order.
The mounting calibration converts vehicle poses to sensor poses. Confirm the
publisher's convention for the chosen export before enabling it. No absent poses
or observed hardware timestamps are fabricated; scan-export timestamps remain
explicitly nominal 20 Hz when that is the available source.

The optional downstream probe requires `downstream: true` and
`label_source_confirmed: true`. The caller confirms native/simulator class meanings
and alignment. Labels are not remapped automatically. The probe is a small point-wise
semantic MLP with fixed training steps, repeated seeds and real/synthetic point-label
budgets. Requested budgets are capped to the common available labeled support
across arms and the actual counts are exported. Development mIoU selects the arm;
only then are final real test features and labels decoded. Without this option,
the primary metric is a clearly named development OT surrogate.

## Controllers and compute

The environment exposes sequential guidance, dropout, scenario and sample-count
controls, discrepancies, remaining budget and previous improvement. SAC uses a
tanh Gaussian actor, replay, twin critics, target networks and learned entropy
temperature. The bandit chooses once per scenario with linear UCB. Existing PPO,
random search, BO and CMA-ES are reused through the same environment.

The isolated `lidarpoc/rl/` library also supplies A2C, REINFORCE, synchronous
n-step actor-critic, DDPG, TD3, DQN, Enhanced DQN, n-step DQN, C51 and a source-style
Rainbow variant. Each has an independent method entry point backed by shared
replay/action/policy modules. The Rainbow variant omits NoisyNet; n-step
actor-critic is synchronous. Newly adapted methods use the top-level `rl` settings,
learn from replay batches capped by available transitions, and export update
counts. Existing PPO/SAC policy formats remain readable. See the README's
[RL method library](../README.md#rl-method-library) for method IDs and settings.

Every generated sample consumes one call, including the initial baseline. Every
experimental sample uses two denoiser passes per diffusion step for equal guidance
cost accounting. The generator stays frozen. A method cannot exceed its call cap;
the elapsed-time cap is checked between actions, so the final action or optimizer
update can exceed it. Actual calls, denoiser evaluations and elapsed time are saved.
Exceeding the time cap prevents the matched-budget RL gate from passing.

Evaluation uses a common fixed scene mixture and one sample per call; scenario
selection/sample count remain training controls. Policy inference may update state
from generated samples against the embedded **training** reference. It never uses
real development/test references as policy observations.

Per-arm costs include training, controller fitting, development/test generation,
optional synthetic training banks and CPU probe fitting. E5 explicitly shares E4's
training cost; it does not imply another E4 training run. GPU hours mean elapsed
allocated GPU count, not kernel-active time or power measurements. Costs per positive
development improvement over E0 are exported only when a positive gain exists.
CPU fixture runs record zero GPU hours and do not measure CUDA throughput.

## Selection, exports and limits

`experiment_plan.json` fixes configuration and reference provenance. All diffusion
weights, policies, label probes, arm selection and the strongest eligible non-RL
comparator freeze in `selection.json` before real test arrays are opened. File-hash
integrity checks inspect test-cache bytes earlier without decoding training inputs.
Repeated-seed results are averaged per independent real session before bootstrap.
RL value requires both development improvement and positive paired test-session
gain against the frozen matched-budget comparator. No optimizer is removed if
that hypothesis fails.
Comparisons are exported per selected method in `gates.rl_comparisons`; the
existing `rl_development_comparison` field retains the SAC comparison. Call/time
cap checks use each method and its selected comparator.

`result.json` includes sensor components, geometry, temporal support, labeled task
results where enabled, label-efficiency curves, compute and decision gates. Artifact
ZIPs include plans, frozen selections/references, scenes, per-seed checkpoints,
controller policies, sample banks, optional probes and license evidence. Generate
accepts an E5 `controller.pt` only with its exact E4 checkpoint and manifest.

A lower transport score is insufficient for success. Missing real pose/task evidence
keeps the decision pending. Geometry failure or no demonstrated task transfer
rejects the tested hypothesis. Native MIT evidence and pinned revisions remain
unchanged; collection rights and the complete dependency/asset audit remain pending.
Full Pandar128 runs, CUDA performance, cross-sensor transfer and production driving
perception quality have not been demonstrated by the synthetic CPU tests.

Mathematical references: [unbalanced scaling](https://arxiv.org/abs/1607.05816),
[Sinkhorn divergences](https://arxiv.org/abs/1810.08278),
[SAC equations](https://spinningup.openai.com/en/latest/algorithms/sac.html).
