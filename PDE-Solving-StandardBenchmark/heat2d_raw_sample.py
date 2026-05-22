"""Raw Heat2D NPZ sample reader."""

import json
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


PARAMETER_NAMES = ("kappa_1", "kappa_2", "q_1", "q_2")


@dataclass(frozen=True)
class Heat2DRawSample:
    """Raw/physical data loaded from one Heat2D FOM NPZ sample."""

    coordinates: np.ndarray
    triangles: np.ndarray
    material_id: np.ndarray
    T: np.ndarray
    kappa_1: float
    kappa_2: float
    q_1: float
    q_2: float
    geometry_metadata_json: str
    mesh_filename: Optional[str]
    sample_id: Union[int, str]
    geometry_id: Optional[int]

    @classmethod
    def from_npz(cls, data, fallback_sample_id=None):
        parameters = _sample_parameters(data)
        return cls(
            coordinates=np.asarray(data["coordinates"], dtype=np.float32),
            triangles=np.asarray(data["triangles"], dtype=np.int64),
            material_id=np.asarray(data["material_id"], dtype=np.int64),
            T=np.asarray(data["T"], dtype=np.float32),
            kappa_1=parameters["kappa_1"],
            kappa_2=parameters["kappa_2"],
            q_1=parameters["q_1"],
            q_2=parameters["q_2"],
            geometry_metadata_json=_string_scalar(data["geometry_metadata_json"])
            if "geometry_metadata_json" in data.files
            else "{}",
            mesh_filename=_string_scalar(data["mesh_filename"]) if "mesh_filename" in data.files else None,
            sample_id=_int_scalar(data["sample_id"]) if "sample_id" in data.files else fallback_sample_id,
            geometry_id=_int_scalar(data["geometry_id"]) if "geometry_id" in data.files else None,
        )

    @property
    def parameters(self):
        return {
            "kappa_1": self.kappa_1,
            "kappa_2": self.kappa_2,
            "q_1": self.q_1,
            "q_2": self.q_2,
        }

    @property
    def geometry_metadata(self):
        try:
            return json.loads(self.geometry_metadata_json)
        except json.JSONDecodeError:
            return {}

    @property
    def shape_type(self):
        metadata = self.geometry_metadata
        return metadata.get("shape", metadata.get("shape_type", "unknown"))

    @property
    def element_kappa(self):
        return np.where(self.material_id == 1, self.kappa_1, self.kappa_2).astype(np.float32)

    @property
    def element_q(self):
        return np.where(self.material_id == 1, self.q_1, self.q_2).astype(np.float32)


def _sample_parameters(data):
    if "param_names" in data.files and "param_values" in data.files:
        names = [_string_scalar(name) for name in np.asarray(data["param_names"])]
        values = np.asarray(data["param_values"], dtype=np.float32)
        params = {name: float(value) for name, value in zip(names, values)}
    else:
        params = {name: float(np.asarray(data[name]).reshape(())) for name in PARAMETER_NAMES if name in data.files}

    missing = [name for name in PARAMETER_NAMES if name not in params]
    if missing:
        raise KeyError(f"Missing Heat2D parameter values: {missing}")
    return params


def _string_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _int_scalar(value):
    return int(np.asarray(value).reshape(()))
