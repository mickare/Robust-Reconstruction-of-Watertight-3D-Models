import os
from dataclasses import dataclass
from typing import Optional, List, Sequence, Iterator

import numpy as np
import plyfile
import plotly.graph_objects as go
import tqdm

from mathlib import quaternion_rotation_matrix


class Transform:
    translation: np.ndarray
    rotation: np.ndarray

    def __init__(self, translation: np.ndarray, rotation: np.ndarray):
        assert translation.shape == (3,)
        assert rotation.shape == (3, 3)
        self.translation = translation
        self.rotation = rotation

    @classmethod
    def from_quat(cls, transl, rot: np.ndarray):
        return cls(transl, quaternion_rotation_matrix(rot))

    @classmethod
    def read(cls, data: Sequence[float]):
        assert len(data) == 7
        d = [float(d) for d in data]
        return cls.from_quat(np.array(d[:3]), np.array(d[3:]))

    def apply(self, points: np.ndarray):
        assert points.shape[1] == 3
        return np.dot((points + self.translation), self.rotation)


class Scan:
    _data: Optional[plyfile.PlyData] = None

    def __init__(self, file: str, trans: Transform):
        self.file = file
        self.trans = trans

    def get(self) -> plyfile.PlyData:
        if self._data is None:
            self._data = plyfile.PlyData.read(self.file)
        return self._data

    def points(self) -> np.ndarray:
        p = self.get()["vertex"]
        pts = np.transpose((p["x"], p["y"], p["z"]))
        return self.trans.apply(pts)

    def scatter(self, transf: Optional[Transform] = None, **kwargs):
        pts = self.points()
        if transf is not None:
            pts = transf.apply(pts)
        # Flip Y/Z
        x, z, y = pts.T
        return go.Scatter3d(x=x, y=y, z=z, **kwargs)


@dataclass
class ScanFolder:
    transform: Transform
    scans: List[Scan]

    @classmethod
    def load_ply_conf(cls, path: str):
        assert os.path.isfile(path)

        base = os.path.dirname(path)

        with open(path, mode='tr') as fp:
            conf = fp.readlines()

        transform: Optional[Transform] = None
        scans: List[Scan] = []

        for line in conf:
            if line.startswith("camera"):
                pass
            elif line.startswith("mesh"):
                l = line.split(" ")
                transform = Transform.read(l[2:])

            elif line.startswith("bmesh"):
                l = line.split(" ")
                scans.append(Scan(os.path.join(base, l[1]), Transform.read(l[2:])))
        return cls(transform, scans)

    def iter_points(self) -> Iterator[np.ndarray]:
        for scan in self.scans:
            yield scan.points()

