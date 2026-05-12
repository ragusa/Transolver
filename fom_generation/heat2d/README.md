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

## Generate a Tiny Dataset

```bash
python -m fom_generation.heat2d.generate_dataset \
  --config fom_generation/heat2d/config_default.json \
  --out data/heat2d_fom_demo \
  --n-geometries 1 \
  --n-params-per-geometry 1 \
  --plot-first
```

The output layout is:

```text
data/heat2d_fom_demo/
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

## Smoke Test

Run:

```bash
python -m fom_generation.heat2d.smoke_test
```

The smoke test generates disk, square, and triangle inclusions, assembles affine components, solves one positive-source problem, checks zero boundary values, confirms both material tags, and checks the free-degree residual.

## Current Limitations

This first pass uses P1 triangular elements, piecewise constant material/source values, one inclusion per geometry, zero outer Dirichlet data, and `.npz` output only. Time dependence, HDF5 export, multiple inclusions, and Transolver dataset adapters are left as later extensions.
