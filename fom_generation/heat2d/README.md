# Heat2D Full-Order Model Generation

This folder contains a small, self-contained generator for steady 2D heat-conduction full-order-model data. It is intended to produce mesh-based samples that can later be adapted for Transolver training, without changing the existing Transolver model or training code.

## Dependencies

Install the local dependencies with:

```bash
pip install -r fom_generation/heat2d/requirements.txt
```

The core packages are `numpy`, `scipy`, `gmsh`, `meshio`, `scikit-fem`, and `matplotlib`. `tqdm` is optional and only adds progress bars.

## Problem

The solver computes the steady equation

```text
-div(kappa(x, y) grad T(x, y)) = q(x, y)
T = 0 on the outer square boundary
```

The square contains one inclusion: a disk, square, or triangle. The inclusion is fully inside the outer square. Gmsh boolean fragmentation is used so the triangular mesh conforms to the material interface.

Physical tags are fixed as:

```text
material_1/background = 1
material_2/inclusion  = 2
outer_boundary        = 101
```

The material interface is internal and receives no boundary condition.

## Affine Decomposition

For each fixed mesh and geometry, the FEM code assembles:

```text
K1 = integral over material_1 of grad(phi_j) dot grad(phi_i)
K2 = integral over material_2 of grad(phi_j) dot grad(phi_i)
F1 = integral over material_1 of phi_i
F2 = integral over material_2 of phi_i
```

Repeated parameter solves then use:

```text
A = kappa_1 K1 + kappa_2 K2
b = q_1 F1 + q_2 F2
```

Homogeneous Dirichlet conditions are applied only to the `outer_boundary` nodes.

## Manufactured-Solution Validation

The validation driver checks the homogeneous single-material case by using the
same conductivity in the background and inclusion. For

```text
T(x, y) = sin(pi x) sin(pi y)
```

and the implemented sign convention `-div(kappa grad T) = q`, the source is

```text
q(x, y) = 2 kappa pi^2 sin(pi x) sin(pi y)
```

Run:

```bash
python -m fom_generation.heat2d.validate_manufactured --mesh-sizes 0.12 0.06
```

The command writes `fom_generation/data/heat2d_validation/manufactured_validation.json` and
prints the L2-like relative error for each mesh.

## Generate a Tiny Dataset

```bash
python -m fom_generation.heat2d.generate_dataset \
  --config fom_generation/heat2d/config_default.json \
  --out fom_generation/data/heat2d_fom_demo \
  --n-geometries 1 \
  --n-params-per-geometry 1 \
  --plot-first
```

The output layout is:

```text
fom_generation/data/heat2d_fom_demo/
  config_used.json
  geometries/
    geom_00000/
      mesh.msh
      mesh_metadata.json
      affine_info.json
      sample_00000.npz
      mesh_materials.png        # if --plot-first is used
      solution.png              # if --plot-first is used
```

Each `.npz` sample contains coordinates, triangle connectivity, element material IDs, nodal temperature `T`, parameter names and values, geometry metadata, mesh filename, sample ID, and geometry ID.

The actual generated sample schema is:

```text
coordinates            (n_nodes, 2)    node coordinates
triangles              (n_elements, 3) triangular connectivity
material_id            (n_elements,)   element physical material tag, 1 or 2
T                      (n_nodes,)      nodal temperature solution
param_names            (4,)            kappa_1, kappa_2, q_1, q_2
param_values           (4,)            parameter values matching param_names
geometry_metadata_json scalar string   Gmsh geometry metadata
mesh_filename          scalar string   mesh path used for the sample
sample_id              scalar int      parameter sample index within geometry
geometry_id            scalar int      geometry index
```

Check and summarize either this layout or an older manifest-based Heat2D dataset with:

```bash
PYTHONPATH=$PWD python -m fom_generation.heat2d.check_dataset --dataset /path/to/dataset
```

## Transolver-Oriented Pilot Dataset

The existing Transolver examples use a few data conventions:

- fixed-grid PDE benchmarks load dense `.mat` or `.npy` tensors with equal
  node counts per sample;
- irregular geometry examples use per-case files plus a `manifest.json`, and
  rely on graph-style or batch-size-one loading for variable node counts.

The randomized heat2d triangular meshes naturally have variable numbers of
nodes and elements, so the pilot uses one compressed `.npz` per sample plus a
manifest. Each sample includes raw FEM arrays and Transolver-oriented aliases:

```text
pos                    (n_nodes, 2)    node coordinates
node_features          (n_nodes, 8)    kappa/q/material/boundary/parameter features
input_feature_names    (8,)            names for node_features columns
target                 (n_nodes, 1)    nodal temperature
coordinates            (n_nodes, 2)    duplicate of pos for explicit FEM naming
triangles              (n_elements, 3) triangular connectivity
element_material_id    (n_elements,)   element physical material tag
nodal_material_id      (n_nodes,)      rounded node material tag for quick checks
edge_index             (2, n_edges)    directed triangle-edge graph, optional
kappa_1, kappa_2       scalar          material diffusion coefficients
q_1, q_2               scalar          piecewise constant source values
shape_type             scalar string   disk, square, or triangle
shape_type_id          scalar int      disk=0, square=1, triangle=2
geometry_metadata_json scalar string   Gmsh geometry metadata
boundary_condition     scalar string   T=0 on outer_boundary
pde_sign_convention    scalar string   -div(kappa grad T)=q
```

Generate the small pilot:

```bash
python -m fom_generation.heat2d.generate_pilot_dataset --n-samples 6
```

Smoke-test loading:

```bash
python -m fom_generation.heat2d.pilot_loader_smoke_test
```

Outputs are written under `fom_generation/data/heat2d_pilot/`. For later
training, variable-size meshes should be handled with batch size 1, a custom
padding/mask collate function, PyG-style batching, approximately fixed mesh
sizes, or resampling to a fixed query set.

## Smoke Test

Run:

```bash
python -m fom_generation.heat2d.smoke_test
```

The smoke test generates disk, square, and triangle inclusions, assembles affine components, solves one positive-source problem, checks zero boundary values, confirms both material tags, and checks the free-degree residual.

## Two-Material Label Validation

Run:

```bash
python -m fom_generation.heat2d.validate_materials
```

This generates one disk, square, and triangle inclusion, verifies that material
tags `1` and `2` both survive import, solves with `kappa_1 != kappa_2` and
`q_1 = q_2 = 1`, and writes one diagnostic material/temperature image per
shape under `fom_generation/data/heat2d_material_validation/`.

## Random Geometry Visual Audit

Run:

```bash
python -m fom_generation.heat2d.validate_geometry_randomization
```

This generates a few reproducible randomized disk, rotated square, and rotated
triangle inclusions, checks that both material tags are nonempty after import,
and writes cellwise material-ID plots under
`fom_generation/data/heat2d_geometry_randomization/<shape>/case_*/material_ids.png`.

## Random Geometry Stress Validation

Run:

```bash
python -m fom_generation.heat2d.validate_geometry_stress --n-geometries 36
```

This generates many randomized two-material geometries without solving the PDE,
checks that both material tags survive import, records any pre-Gmsh rejected
geometry samples, and saves a representative subset of cellwise material-ID
plots under `fom_generation/data/heat2d_geometry_stress/`.

## Current Limitations

This first pass uses P1 triangular elements, piecewise constant material/source values, one inclusion per geometry, zero outer Dirichlet data, and `.npz` output only. Time dependence, HDF5 export, multiple inclusions, and Transolver dataset adapters are left as later extensions.
