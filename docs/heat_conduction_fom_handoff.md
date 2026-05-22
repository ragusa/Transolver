# Heat-Conduction FOM Handoff

## 1. Purpose

This repository now contains a self-contained workflow for generating full-order-model (FOM) solutions for 2D steady heat conduction on triangular Gmsh meshes.

The workflow lives under:

```text
fom_generation/heat2d/
```

Its purpose is to create small, validated mesh-based datasets that can later be used to train or evaluate Transolver. The Transolver model and training code have not yet been adapted.

## 2. PDE Form And Sign Convention

Implemented PDE:

```text
-div(kappa(x, y) grad T(x, y)) = q(x, y) in Omega
T = 0 on the outer square boundary
```

Weak form assembled by scikit-fem:

```text
integral kappa grad(T) . grad(v) = integral q v
```

Boundary convention:

- Homogeneous Dirichlet condition only on the outer square boundary.
- Internal material interfaces are not boundary conditions.
- Default boundary value is `T = 0`.

The sign convention was verified by a manufactured solution:

```text
T(x, y) = sin(pi x) sin(pi y)
q(x, y) = 2 kappa pi^2 sin(pi x) sin(pi y)
```

## 3. Geometry Workflow

Domain:

- Outer square domain, default `[0, 1] x [0, 1]`.
- One internal inclusion.

Supported inclusion shapes:

- `disk`
- `square`
- `triangle`

Randomization supports:

- inclusion center,
- inclusion size,
- rotation angle for square and triangle inclusions.

Validity checks:

- configurable safety margin from the outer boundary,
- reject/resample inclusions that are too small, too large, degenerate, or outside the allowed square interior,
- square and triangle rotations are sampled away from axis-aligned angles for random audits.

Mesh generation:

- Uses the Gmsh Python API.
- Uses OCC geometry.
- Uses boolean fragmentation so the mesh conforms to the internal material interface.
- Uses physical groups for material and boundary tags.

## 4. Material Conventions

Physical tags:

```text
material_1/background = 1
material_2/inclusion  = 2
outer_boundary        = 101
```

Material assignment:

- After Gmsh fragmentation, surfaces are classified by inclusion geometry/area.
- Element material IDs are imported from Gmsh physical tags using `meshio`.
- `element_material_id` is stored per triangle.
- `nodal_material_id` is derived from neighboring element labels for diagnostics and possible feature engineering.

Material parameters:

```text
kappa_1 = background diffusion coefficient
kappa_2 = inclusion diffusion coefficient
q_1     = background source value
q_2     = inclusion source value
```

Affine solve convention:

```text
A = kappa_1 K1 + kappa_2 K2
b = q_1 F1 + q_2 F2
```

## 5. Boundary-Condition And Source Conventions

Dirichlet treatment:

- Boundary dofs are read from Gmsh physical group `outer_boundary = 101`.
- The condensed/free-dof system is solved with boundary nodes fixed to zero.
- Internal material interfaces are unconstrained.

Source representation:

- Piecewise constant source values `q_1`, `q_2` are used for normal dataset generation.
- Callable source terms `q(x, y)` are also supported for validation.

Validation/pilot source choices:

- Manufactured validation uses callable `q(x, y) = 2 kappa pi^2 sin(pi x) sin(pi y)`.
- Pilot dataset uses sampled piecewise constant `q_1`, `q_2`.

## 6. Validation Tests That Passed

Run commands from the repository root. The local tested interpreter was:

```powershell
& 'C:\ProgramData\anaconda3\python.exe'
```

Smoke test:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m fom_generation.heat2d.smoke_test
```

Checks:

- disk, square, and triangle meshes generate,
- both material tags exist,
- affine components assemble,
- one positive-source solve succeeds,
- outer boundary dofs are zero,
- residual is small.

Expected success:

```text
heat2d smoke test passed
```

Manufactured-solution validation:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m fom_generation.heat2d.validation.manufactured --mesh-sizes 0.12 0.06 0.03
```

Checks:

- single-material behavior by setting same material properties in both regions,
- correct PDE sign convention,
- L2-like error decreases under mesh refinement.

Observed success criterion:

- relative L2-like error decreased from coarse to fine meshes.

Two-material geometry/material validation:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m fom_generation.heat2d.validation.materials
```

Checks:

- disk, square, and triangle inclusion labels survive Gmsh/meshio import,
- both material regions are nonempty,
- solve with `kappa_1 != kappa_2` and `q_1 = q_2 = 1`,
- diagnostic material/temperature plots are saved.

Randomized geometry stress test:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m fom_generation.heat2d.validation.geometry_stress --n-geometries 36 --seed 20260512 --mesh-size 0.11 --safety-margin 0.06 --plots-per-shape 3
```

Checks:

- many random disk/square/triangle geometries,
- random centers, sizes, and rotations,
- pre-Gmsh validity checks,
- material tags are preserved and nonempty,
- representative material-ID plots are saved.

Observed success:

```text
Geometries passed: 36
All cases passed: True
```

Lightweight pytest checks:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m pytest tests/heat2d -q
```

Checks:

- synthetic manifest/sample data loads,
- required `.npz` fields exist,
- shapes are consistent,
- solution values are finite,
- both materials exist in every sample,
- disk, square, and triangle samples are present,
- dataset summary checks work on temporary generated samples.

Expected success:

```text
2 passed
```

## 7. Dataset Generation

Pilot dataset command:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m fom_generation.heat2d.cli.generate_pilot_dataset --out fom_generation/data/heat2d_pilot --n-samples 6 --seed 424242 --mesh-size 0.11 --safety-margin 0.06 --plots-per-shape 1
```

Output directory:

```text
fom_generation/data/heat2d_pilot/
```

Storage convention:

- one compressed `.npz` per sample,
- `manifest.json` at dataset root,
- train/test split recorded in the manifest,
- diagnostic plots saved under `plots/`.

Pilot layout:

```text
fom_generation/data/heat2d_pilot/
  manifest.json
  samples/
    sample_00000_square/
      mesh.msh
      mesh_metadata.json
      sample.npz
    ...
  plots/
    sample_00000_square/
      material_ids.png
      temperature.png
    ...
```

## 8. Current Dataset Schema

Each sample `.npz` contains exactly these intended fields:

```text
pos                    (n_nodes, 2)
node_features          (n_nodes, 8)
input_feature_names    (8,)
target                 (n_nodes, 1)
coordinates            (n_nodes, 2)
triangles              (n_elements, 3)
element_material_id    (n_elements,)
nodal_material_id      (n_nodes,)
edge_index             (2, n_edges)
kappa_1                scalar
kappa_2                scalar
q_1                    scalar
q_2                    scalar
shape_type             scalar string
shape_type_id          scalar int
geometry_metadata_json scalar string
boundary_condition     scalar string
pde_sign_convention    scalar string
```

Field meanings:

- `pos`: node coordinates. This is the first candidate position tensor for Transolver.
- `coordinates`: duplicate of `pos`, kept for explicit FEM naming.
- `node_features`: ML-oriented node input features. This should be the first candidate input tensor for Transolver.
- `input_feature_names`: names of the `node_features` columns.
- `target`: nodal temperature solution `T`.
- `triangles`: triangular element connectivity.
- `element_material_id`: material tag per triangle.
- `nodal_material_id`: node-level material label derived from adjacent elements.
- `edge_index`: directed graph edges derived from triangle connectivity.
- `kappa_1`, `kappa_2`: scalar diffusion coefficients for background and inclusion.
- `q_1`, `q_2`: scalar source values for background and inclusion.
- `shape_type`: inclusion shape string.
- `shape_type_id`: integer shape ID, with `disk=0`, `square=1`, `triangle=2`.
- `geometry_metadata_json`: JSON string with inclusion geometry parameters and Gmsh physical tag metadata.
- `boundary_condition`: string metadata, currently `T=0 on outer_boundary`.
- `pde_sign_convention`: string metadata, currently `-div(kappa grad T)=q`.

`node_features` columns:

```text
kappa_node
q_node
material_2_fraction
outer_boundary_mask
kappa_1
kappa_2
q_1
q_2
```

Column meanings:

- `kappa_node`: nodewise average of adjacent element diffusion values.
- `q_node`: nodewise average of adjacent element source values.
- `material_2_fraction`: fraction of adjacent elements belonging to inclusion material.
- `outer_boundary_mask`: 1 for outer Dirichlet boundary nodes, otherwise 0.
- `kappa_1`, `kappa_2`, `q_1`, `q_2`: sample-level scalars repeated at every node.

Topology/diagnostic note:

- `triangles` and `edge_index` preserve mesh/topology information.
- `nodal_material_id` and `element_material_id` are included for diagnostics, plotting, and possible feature engineering.
- `geometry_metadata_json` stores inclusion center, size, rotation, shape, mesh size, safety margin, and physical tags.

## 9. Visualization Utilities

Utilities are in:

```text
fom_generation/heat2d/visualize.py
```

Functions:

- `plot_mesh_with_materials(mesh_data, ax=None)`
- `plot_solution(mesh_data, T, ax=None)`

Diagnostic pilot plots are saved under:

```text
fom_generation/data/heat2d_pilot/plots/<sample_name>/material_ids.png
fom_generation/data/heat2d_pilot/plots/<sample_name>/temperature.png
```

Other validation plots are saved under:

```text
fom_generation/data/heat2d_material_validation/
fom_generation/data/heat2d_geometry_randomization/
fom_generation/data/heat2d_geometry_stress/
```

## 10. Known Limitations And Design Choices

- Meshes can have variable numbers of nodes and elements.
- Existing fixed-grid Transolver benchmark loaders are not directly compatible with variable-size triangular meshes.
- Current heat2d samples are nodal/point-cloud-like samples with mesh connectivity included.
- The pilot dataset is not yet guaranteed to match Transolver's current loaders without an adapter.
- Batching may require one of:
  - batch size 1,
  - padding plus masks,
  - point-cloud/PyG-style batching,
  - a custom collate function,
  - resampling FOM solutions onto a fixed set of query points.
- Current pilot data use P1 FEM, one inclusion per square, zero outer Dirichlet data, and piecewise constant material/source values.
- This handoff documents the implemented pilot format. Do not replace the format during initial Transolver integration unless a concrete loader issue requires an extension.

## 11. Recommended Next Steps For Fresh Transolver Session

1. Read this handoff file.
2. Inspect existing Transolver dataset loaders and model call signatures:
   - `PDE-Solving-StandardBenchmark/exp_*.py`
   - `PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py`
   - `Airfoil-Design-AirfRANS/dataset/dataset.py`
   - `Car-Design-ShapeNetCar/dataset/dataset.py`
3. Compare the implemented `.npz` schema with the expected Transolver inputs.
4. Decide whether `node_features` should be passed as `fx` directly.
5. Decide how to handle variable `n_nodes`.
6. Implement a heat-conduction dataset loader for `fom_generation/data/heat2d_pilot/manifest.json`.
7. Implement batching/collation.
8. Run a tiny overfitting test on 1 to 3 samples.
9. Run a first small train/validation experiment.
10. Only after that, generate a larger FOM dataset.

## 12. Exact Fresh-Session Prompt

Copy/paste this prompt into the next Codex session:

```text
We are starting the Transolver-integration phase for the 2D heat-conduction FOM dataset.

First, read docs/heat_conduction_fom_handoff.md carefully. Then inspect the existing Transolver dataset loaders, model call signatures, and training scripts before modifying files.

The pilot heat2d dataset format has already been chosen and implemented. Do not propose a replacement format at the start. Use the implemented one-sample-per-.npz plus manifest format under fom_generation/data/heat2d_pilot/.

Your first task is to propose an integration plan for adapting Transolver to this pilot dataset. The plan should address:
- which existing loader pattern is closest,
- how to map pos, node_features, and target into the Transolver model,
- how to handle variable n_nodes,
- what batching or collate strategy to use,
- what minimal overfitting test to run first.

After proposing the plan, wait for confirmation before editing model/training code.
```
