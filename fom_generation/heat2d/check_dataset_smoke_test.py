"""Smoke test for Heat2D dataset checking without a manifest."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .cli.check_dataset import check_dataset


def run_smoke_test():
    with TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp)
        geom_dir = dataset_dir / "geometries" / "geom_00000"
        geom_dir.mkdir(parents=True)
        np.savez_compressed(
            geom_dir / "sample_00000.npz",
            coordinates=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            triangles=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
            material_id=np.array([1, 2], dtype=np.int64),
            T=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            param_names=np.array(["kappa_1", "kappa_2", "q_1", "q_2"]),
            param_values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            geometry_metadata_json=np.array("{}"),
            mesh_filename=np.array("mesh.msh"),
            sample_id=np.array(0, dtype=np.int64),
            geometry_id=np.array(0, dtype=np.int64),
        )

        summary = check_dataset(dataset_dir, write_summary=False)

    assert summary["source"] == "discovered"
    assert summary["num_geometry_dirs"] == 1
    assert summary["num_sample_files"] == 1
    assert summary["solution_field"] == "T"
    assert summary["temperature_stats"] == {"min": 0.0, "mean": 1.5, "max": 3.0}
    assert summary["all_T_finite"]
    assert summary["all_coordinates_finite"]
    assert summary["material_tags_1_and_2_present"]
    assert summary["all_samples_have_material_tags_1_and_2"]
    assert summary["parameter_ranges"]["kappa_2"] == {"min": 2.0, "mean": 2.0, "max": 2.0}
    print("Heat2D check_dataset smoke test passed")


if __name__ == "__main__":
    run_smoke_test()
