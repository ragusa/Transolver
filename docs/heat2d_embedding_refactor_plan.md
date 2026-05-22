# Heat2D Embedding Refactor Plan

## Purpose

This document audits the current Heat2D FOM-to-Transolver path and proposes a
least-disruptive refactor that separates:

1. raw FOM snapshot storage,
2. derived geometric and physical features,
3. Transolver embedding construction.

The goal is to preserve the current working training path while making feature
sets explicit and easier to ablate.

## Current Data Generation

### 1. Where Heat2D FOM Samples Are Generated

The raw FOM generator lives in:

```text
fom_generation/heat2d/dataset.py
```

The current repository no longer keeps generator command scripts directly under
`fom_generation/heat2d/`. The supported command modules live under
`fom_generation/heat2d/cli/`; the reusable generation implementation remains in
`dataset.py`.

The canonical CLI entry point for the raw generator is:

```text
python -m fom_generation.heat2d.cli.generate_dataset
```

The generator flow is:

- load configuration through `fom_generation.heat2d.config`;
- create one mesh per geometry with `generate_two_material_square_mesh`;
- load Gmsh physical tags with `load_mesh_and_tags`;
- assemble affine FEM operators with `assemble_affine_components`;
- solve parameter samples with `solve_for_parameters`;
- write one compressed `.npz` sample per parameter case.

There are also Transolver-oriented generator commands:

```text
python -m fom_generation.heat2d.cli.generate_pilot_dataset
python -m fom_generation.heat2d.cli.generate_balanced_dataset
```

Those currently write samples with raw arrays plus prebuilt Transolver-oriented
aliases/features. They are useful for current training, but they mix raw FOM
storage with embedding design.

### 2. Files Written By The FOM Generator

The raw generator writes under the selected output directory, commonly:

```text
fom_generation/data/heat2d_fom_demo/
```

The layout is:

```text
config_used.json
geometries/
  geom_00000/
    mesh.msh
    mesh_metadata.json
    affine_info.json
    sample_00000.npz
    mesh_materials.png      # only when --plot-first is used
    solution.png            # only when --plot-first is used
```

The pilot and balanced generators write a manifest-style layout, commonly:

```text
manifest.json
samples/
  sample_00000_<shape>/
    mesh.msh
    mesh_metadata.json
    sample.npz
plots/
  sample_00000_<shape>/
    material_ids.png
    temperature.png
```

The output roots remain generated data and should stay under
`fom_generation/data/` or another user-selected output path.

### 3. Current NPZ Sample Format

The raw FOM `.npz` samples written by `fom_generation/heat2d/dataset.py`
contain:

```text
coordinates             shape=(N_node, 2)
triangles               shape=(N_elem, 3)
material_id             shape=(N_elem,)
T                       shape=(N_node,)
param_names             shape=(4,)
param_values            shape=(4,)
geometry_metadata_json  scalar string
mesh_filename           scalar string
sample_id               scalar int
geometry_id             scalar int
```

The current pilot/balanced `.npz` samples written by the Transolver-oriented
generators contain additional fields, including:

```text
pos
node_features
input_feature_names
target
coordinates
triangles
element_material_id
nodal_material_id
edge_index
kappa_1
kappa_2
q_1
q_2
shape_type
shape_type_id
geometry_metadata_json
mesh_filename
sample_id
boundary_condition
pde_sign_convention
```

That richer format is convenient for the current training path, but it embeds a
specific feature choice directly into generated dataset files.

## Current Training Path

### 4. Current Heat2D Dataset Loader

The current loader is:

```text
PDE-Solving-StandardBenchmark/heat2d_dataset.py
```

The main class is `Heat2DDataset`. It supports two sample schemas:

- raw FOM samples discovered under `geometries/geom_*/sample_*.npz`;
- manifest-style samples discovered through `manifest.json`.

The collate function is:

```text
heat2d_batch_size_one_collate
```

It wraps one variable-size mesh sample into batch-size-one tensors with shapes
compatible with the current irregular-mesh Transolver model.

### 5. Where Node Features Are Currently Constructed

For raw FOM samples, node features are currently constructed in:

```text
Heat2DDataset._load_fom_sample
```

The helper functions used there are:

```text
_sample_parameters
_nodal_material_2_fraction
_outer_boundary_mask
_triangle_edge_index
```

For manifest-style pilot/balanced samples, the loader reads `node_features`
directly from the `.npz` if present:

```text
Heat2DDataset._load_manifest_sample
```

Pilot/balanced `node_features` are created during generation in:

```text
fom_generation/heat2d/cli/generate_pilot_dataset.py
```

and reused by:

```text
fom_generation/heat2d/cli/generate_balanced_dataset.py
```

The key helper is `_build_node_features`.

### 6. Current Feature Vector Passed To Transolver

For raw FOM samples, the current default feature vector is:

```text
fx = [
  material_2_fraction,
  kappa_1,
  kappa_2,
  q_1,
  q_2,
]
```

If `--include-boundary-mask` is used, `outer_boundary_mask` is appended.

For pilot/balanced manifest samples, the generated feature vector is:

```text
fx = [
  kappa_node,
  q_node,
  material_2_fraction,
  outer_boundary_mask,
  kappa_1,
  kappa_2,
  q_1,
  q_2,
]
```

This means the effective embedding can differ depending on whether training
loads raw FOM samples or manifest-style generated samples.

### 7. Training Entry Point

The current training entry point is:

```text
PDE-Solving-StandardBenchmark/exp_heat2d.py
```

The script imports:

```text
from heat2d_dataset import Heat2DDataset, heat2d_batch_size_one_collate
from model.Transolver_Irregular_Mesh import Model
```

It supports modes:

```text
smoke
overfit
train
eval
```

### 8. How Training Consumes `x`, `fx`, And `y`

The dataset returns dictionaries containing:

```text
pos            node coordinates
node_features  feature vector
target         nodal temperature
triangles      mesh connectivity for plotting/metrics
```

The collate function maps one sample to:

```text
batch["pos"]            shape=(1, N_node, 2)
batch["node_features"]  shape=(1, N_node, F)
batch["target"]         shape=(1, N_node, 1)
```

`exp_heat2d.py` then normalizes the batch in `normalize_batch`:

```text
pos = batch["pos"]
fx  = batch["node_features"]
y   = batch["target"]
```

The model call is:

```text
pred = model(pos, fx=fx)
```

So the current Transolver path is:

```text
x  = coordinates / pos
fx = node_features
y  = T / target
```

### 9. Current Coupling Between Raw FOM Data And Embedding Design

The current implementation has several couplings:

- `Heat2DDataset` both reads raw `.npz` data and decides which node features to
  build.
- Raw FOM samples and manifest-style samples can produce different `fx`
  definitions.
- Pilot/balanced generators write `node_features` into the sample file, so a
  generated dataset can freeze a particular embedding design.
- `Heat2DDataset.num_input_features` and `exp_heat2d.py` model construction
  depend directly on whichever feature vector the loader returns.
- `compute_stats` normalizes `node_features` without an explicit feature-set
  identity beyond `input_feature_names`.
- The `--include-boundary-mask` flag changes the feature vector as a loader
  detail rather than as a named embedding choice.
- Derived quantities such as `material_2_fraction`, nodal kappa, nodal q,
  boundary masks, and graph edges are spread between the generator and loader.

This coupling makes feature ablations possible but awkward: changing features
can require editing generation code, loader logic, training defaults, and
metadata expectations at the same time.

## Desired Architecture

### Tier 1: Raw FOM Sample Reader

Create a small reader abstraction for existing NPZ samples. It should not
require HDF5 yet.

Recommended object:

```text
Heat2DRawSample
```

It should expose:

```text
coordinates
triangles
material_id
T
kappa_1
kappa_2
q_1
q_2
geometry_metadata_json
sample_id
geometry_id
```

It may also expose raw/physical derived arrays:

```text
element_kappa = where(material_id == 1, kappa_1, kappa_2)
element_q     = where(material_id == 1, q_1, q_2)
```

Those should be treated as physical sample data, not a Transolver embedding.

The reader should support both current raw FOM parameter formats:

- `param_names` plus `param_values`;
- scalar `kappa_1`, `kappa_2`, `q_1`, `q_2` fields when present.

For Phase A, manifest-style samples are intentionally left on the existing
loader path and continue to read precomputed `node_features` directly from the
NPZ files. Unifying raw-FOM and manifest-style reading is deferred to Phase B or
later.

### Tier 2: Derived Feature Utilities And Cache

Add dependency-light utilities for derived features, initially in memory.

Candidate module:

```text
PDE-Solving-StandardBenchmark/heat2d_features.py
```

or, if the code is intended to be shared outside the benchmark training path:

```text
fom_generation/heat2d/features.py
```

Initial derived features:

```text
nodal_material_2_fraction
nodal_kappa_arithmetic
nodal_kappa_harmonic
nodal_q
outer_boundary_mask
distance_to_outer_boundary
element_centroids
element_areas
```

Later features:

```text
interface_distance
signed distance to inclusion boundary
local mesh size / nodal area weights
element-to-node aggregation variants
```

The first implementation should compute features in memory from a
`Heat2DRawSample`. A persistent cache is not needed yet. If repeated feature
construction becomes expensive at larger scale, add a cache behind the same
function interface.

### Tier 3: Embedding Builders

Introduce an `EmbeddingBuilder`-style abstraction that converts a raw sample and
derived feature utilities into the tensors consumed by Transolver.

Suggested interface:

```text
class EmbeddingBuilder:
    name: str
    feature_names: tuple[str, ...]

    def build(self, sample) -> dict:
        return {
            "pos": ...,
            "node_features": ...,
            "target": ...,
            "triangles": ...,
            "element_material_id": ...,
            "metadata": ...,
        }
```

The current model should continue to receive node-based tensors:

```text
x  shape=(N_node, 2)
fx shape=(N_node, n_features)
y  shape=(N_node, 1)
```

Do not switch to element-token inputs yet. Element tokens would require model
and collate changes, decisions about target placement, and new plotting/metric
logic. Element-level features should be computed and exposed as diagnostics or
future inputs, not consumed by the current model path in this refactor.

Phase A builders:

#### BasicEmbeddingBuilder

Preserves the current raw-FOM training feature set:

```text
material_2_fraction
kappa_1
kappa_2
q_1
q_2
```

This should be the default at first to preserve existing behavior.

#### BasicWithBoundaryMaskEmbeddingBuilder

Preserves the current raw-FOM `--include-boundary-mask` behavior:

```text
material_2_fraction
kappa_1
kappa_2
q_1
q_2
outer_boundary_mask
```

This is not a new experimental feature set. It is a compatibility builder for
the existing flag.

Phase B builders should be organized as a controlled 2 x 2 ablation, not as a
linear chain of cumulative feature additions:

```text
derived physical node fields absent/present
explicit geometric distance fields absent/present
```

#### PhysicalEmbeddingBuilder

Derived physical node fields are present. Explicit geometric distance fields
are absent:

```text
material_2_fraction
kappa_1
kappa_2
q_1
q_2
kappa_node_arithmetic
kappa_node_harmonic
q_node
outer_boundary_mask
```

#### GeometryAwareEmbeddingBuilder

Explicit geometric distance fields are present. Derived local physical fields
such as `kappa_node_arithmetic`, `kappa_node_harmonic`, and `q_node` are absent:

```text
material_2_fraction
kappa_1
kappa_2
q_1
q_2
outer_boundary_mask
signed_distance_to_interface
distance_to_outer_boundary
```

This builder is important because it diagnoses how much improvement comes from
geometric localization alone.

#### PhysicalPlusEmbeddingBuilder

Derived physical node fields and explicit geometric distance fields are both
present:

```text
material_2_fraction
kappa_1
kappa_2
q_1
q_2
kappa_node_arithmetic
kappa_node_harmonic
q_node
outer_boundary_mask
signed_distance_to_interface
distance_to_outer_boundary
```

If exact preservation of the current pilot/balanced generated feature order is
important, provide a named compatibility builder such as
`GeneratedPilotEmbeddingBuilder` rather than hiding that behavior in the loader.

## Least-Disruptive Refactor Plan

### Phase A: Reader And Builder Abstraction With No Behavior Change

Add a raw sample reader and embedding builder abstraction while preserving NPZ
storage and current training behavior.

Tasks:

1. Add a `Heat2DRawSample` reader for current `.npz` files.
2. Move parameter parsing out of `Heat2DDataset` into the reader.
3. Move current raw-FOM feature construction into `BasicEmbeddingBuilder`.
4. Keep `Heat2DDataset` returning the same dictionary keys:
   - `pos`
   - `node_features`
   - `target`
   - `triangles`
   - `element_material_id`
   - `nodal_material_id`
   - `edge_index`
   - metadata fields
5. Keep `exp_heat2d.py` behavior unchanged:
   - `pos, fx, y = normalize_batch(...)`
   - `pred = model(pos, fx=fx)`
6. Add focused tests for:
   - reading raw FOM NPZ samples;
   - `BasicEmbeddingBuilder` and `BasicWithBoundaryMaskEmbeddingBuilder`
     numerical equivalence to the legacy raw-FOM feature construction;
   - current `Heat2DDataset` output compatibility for both raw-FOM variants,
     including feature names and feature order.

Default behavior should remain equivalent to the current raw FOM path:

```text
--feature-set basic
```

or no flag yet, if adding the flag is deferred to Phase B.

### Phase B: Controlled 2 x 2 Feature Ablation

Add an explicit CLI option to `PDE-Solving-StandardBenchmark/exp_heat2d.py`:

```text
--feature-set basic
--feature-set basic_with_boundary_mask
--feature-set physical
--feature-set geometry_aware
--feature-set physical_plus
```

Phase B should be framed as a controlled 2 x 2 ablation:

```text
                         geometric distance features absent   geometric distance features present
physical features absent BasicEmbeddingBuilder                GeometryAwareEmbeddingBuilder
physical features present PhysicalEmbeddingBuilder             PhysicalPlusEmbeddingBuilder
```

`BasicWithBoundaryMaskEmbeddingBuilder` remains a compatibility builder for the
existing `--include-boundary-mask` behavior. It should not be counted as the
physical-present/geometric-present cell because the boundary mask is an existing
loader option, not the intended interface-localization feature.

Implementation tasks:

1. Pass `feature_set` from `exp_heat2d.py` into `Heat2DDataset`.
2. Select an embedding builder by name.
3. Set `dataset.input_feature_names` from the builder.
4. Set `dataset.num_input_features` from the selected builder output.
5. Keep model construction based on `train_dataset.num_input_features`.
6. Save `feature_set` and `input_feature_names` in run config/summary JSON.
7. Validate that checkpoint eval restores or validates the feature set used for
   training.

Recommended default:

```text
--feature-set basic
```

This preserves the current raw-FOM training path. If current experiments mainly
use pilot/balanced manifest samples with eight precomputed features, add an
explicit compatibility option and document it rather than making it implicit.

#### Phase B Feature Sets

1. `BasicEmbeddingBuilder`

   No derived physical node fields. No explicit geometric distance fields.

   ```text
   material_2_fraction
   kappa_1
   kappa_2
   q_1
   q_2
   ```

2. `BasicWithBoundaryMaskEmbeddingBuilder`

   Exists to preserve the current `--include-boundary-mask` behavior.

   ```text
   material_2_fraction
   kappa_1
   kappa_2
   q_1
   q_2
   outer_boundary_mask
   ```

3. `PhysicalEmbeddingBuilder`

   Derived physical node fields are present. Explicit geometric distance fields
   are absent.

   ```text
   material_2_fraction
   kappa_1
   kappa_2
   q_1
   q_2
   kappa_node_arithmetic
   kappa_node_harmonic
   q_node
   outer_boundary_mask
   ```

4. `GeometryAwareEmbeddingBuilder`

   Explicit geometric distance fields are present. Derived local physical fields
   such as `kappa_node_arithmetic`, `kappa_node_harmonic`, and `q_node` are
   absent.

   ```text
   material_2_fraction
   kappa_1
   kappa_2
   q_1
   q_2
   outer_boundary_mask
   signed_distance_to_interface
   distance_to_outer_boundary
   ```

5. `PhysicalPlusEmbeddingBuilder`

   Derived physical node fields and explicit geometric distance fields are both
   present.

   ```text
   material_2_fraction
   kappa_1
   kappa_2
   q_1
   q_2
   kappa_node_arithmetic
   kappa_node_harmonic
   q_node
   outer_boundary_mask
   signed_distance_to_interface
   distance_to_outer_boundary
   ```

The purpose of this structure is experimental clarity. It separately measures
the marginal value of derived physical features and geometric localization
features.

#### Signed Distance To Interface

Node-level signed distance to the material interface is explicitly part of
Phase B through `GeometryAwareEmbeddingBuilder` and `PhysicalPlusEmbeddingBuilder`.
Do not defer it to an undefined later phase. Without this feature, the ablation
cannot answer whether geometry-aware embeddings help.

Use this sign convention:

```text
signed_distance_to_interface < 0 inside the inclusion / material 2
signed_distance_to_interface = 0 on the material interface
signed_distance_to_interface > 0 in the background / material 1
```

Normalize distances by the outer square length if useful so the feature scale is
O(1).

Before implementing signed distance, inspect `geometry_metadata_json` and verify
which geometric parameters are stored for disk, square, and triangle inclusions.
If the metadata are insufficient to reconstruct analytic signed distance,
document exactly what is missing and propose the smallest generator metadata
addition. Do not silently approximate missing signed distance values with zeros
or placeholders.

Expected analytic SDF strategies:

1. Disk:

   ```text
   sdf = sqrt((x - cx)^2 + (y - cy)^2) - r
   ```

   where `(cx, cy)` is the disk center and `r` is the radius.

2. Square:

   Transform node coordinates into the local rotated-square frame and use the
   standard signed-distance function for a square/rectangle.

3. Triangle:

   Use the signed minimum distance to the three finite triangle edge segments.
   The distance must be to each finite line segment, not to the infinite line.
   When projecting a node onto an edge, clamp the projection parameter to the
   segment endpoints before computing the distance. This matters near triangle
   corners, where the closest point may be a vertex rather than an interior point
   of an edge. Determine the sign separately using a robust point-in-triangle
   test.

### Phase C: Feature Ablation Slurm Commands For Vision

Add Slurm commands/scripts for feature-set ablations on Vision.

Recommended controlled experiment matrix:

```text
basic            physical absent   geometric distance absent
geometry_aware   physical absent   geometric distance present
physical         physical present  geometric distance absent
physical_plus    physical present  geometric distance present
```

Run `basic_with_boundary_mask` only when comparing to historical
`--include-boundary-mask` runs or checking compatibility with the existing flag.

Keep all other training settings fixed:

```text
dataset
train/val/test split
seed
model size
epochs
learning rate
batch size
normalization
```

Each run should write a distinct output directory, for example:

```text
results/heat2d_feature_ablation/basic
results/heat2d_feature_ablation/geometry_aware
results/heat2d_feature_ablation/physical
results/heat2d_feature_ablation/physical_plus
```

The Slurm scripts should record:

```text
feature_set
input_feature_names
dataset path
git commit
hostname / job id
```

### Phase D: HDF5 And Persistent Feature Cache Only If Needed

Do not introduce HDF5 or a persistent feature cache immediately.

Consider HDF5 only if:

- NPZ-per-sample metadata overhead becomes a bottleneck;
- filesystem pressure becomes significant at larger sample counts;
- repeated feature construction materially slows training;
- multi-worker data loading becomes I/O-bound.

If caching becomes necessary, keep the same reader/builder interfaces and add a
cache implementation behind them. The cache should store derived features with
metadata that includes:

```text
feature_set
feature_names
source sample path
source sample checksum or mtime/size
code version / schema version
```

## Recommended File-Level Changes

Do not implement these in the audit step. This is the proposed future patch
shape.

Candidate new files:

```text
PDE-Solving-StandardBenchmark/heat2d_raw_sample.py
PDE-Solving-StandardBenchmark/heat2d_features.py
PDE-Solving-StandardBenchmark/heat2d_embeddings.py
```

Candidate edits:

```text
PDE-Solving-StandardBenchmark/heat2d_dataset.py
PDE-Solving-StandardBenchmark/exp_heat2d.py
PDE-Solving-StandardBenchmark/README_HEAT2D.md
tests/heat2d/
```

Keep generated data untouched:

```text
fom_generation/data/
```

## Risks And Guardrails

- Preserve the current `x`, `fx`, `y` contract until a separate model-level
  refactor is planned.
- Do not remove support for current NPZ files.
- Avoid storing a single hard-coded `node_features` meaning inside generated
  raw FOM samples.
- Keep feature names explicit and saved with each training run.
- Keep batch-size-one behavior until padded batching or graph-style batching is
  designed separately.
- Keep element-token experiments out of this refactor unless model changes are
  explicitly scheduled.

## Summary

The current training path works, but `Heat2DDataset` and the pilot/balanced
generators mix raw sample reading, derived feature computation, and Transolver
embedding choices. The least disruptive path is to first introduce a raw sample
reader plus `BasicEmbeddingBuilder` and `BasicWithBoundaryMaskEmbeddingBuilder`
that exactly preserve current raw-FOM behavior, then add named feature sets
behind a `--feature-set` flag. Phase B should be a controlled 2 x 2 ablation
that explicitly includes signed distance to the material interface. HDF5 and
persistent caches should wait until scaling measurements justify them.
