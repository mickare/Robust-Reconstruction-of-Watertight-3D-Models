import os
from dataclasses import dataclass
from typing import Optional, List, Sequence, Iterator

import numpy as np
import plyfile
import plotly.graph_objects as go
import tqdm

from mathlib import quaternion_rotation_matrix
from model import ModelLoader


class Transform:
    translation: np.ndarray
    rotation: np.ndarray

    def __init__(self, translation: np.ndarray, rotation: np.ndarray):
        assert translation.shape == (3,)
        assert rotation.shape == (3, 3)
        self.translation = translation
        self.rotation = rotation

    @classmethod
    def identity(cls) -> "Transform":
        return cls(np.array([0, 0, 0]), np.identity(3))

    @classmethod
    def from_quat(cls, transl, rot: np.ndarray) -> "Transform":
        return cls(transl, quaternion_rotation_matrix(rot))

    @classmethod
    def read(cls, data: Sequence[float]) -> "Transform":
        assert len(data) == 7
        d = [float(d) for d in data]
        return cls.from_quat(np.array(d[:3]), np.array(d[3:]))

    def apply(self, points: np.ndarray) -> np.ndarray:
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

        transform: Optional[Transform] = Transform.identity()
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


class PlyModelLoader(ModelLoader):
    def load(self, path: str) -> np.ndarray:
        scan = ScanFolder.load_ply_conf(path)
        return scan.transform.apply(np.concatenate([s.points() for s in scan.scans]))


def plot_dragon():
    import plotly.graph_objects as go
    import tqdm

    scans = ScanFolder.load_ply_conf("models/dragon_stand/dragonStandRight.conf")

    fig = go.Figure(
        data=[
            s.scatter(transf=scans.transform, mode='markers', marker=dict(size=0.5))
            for s in tqdm.tqdm(scans.scans, desc="Load files")
        ]
    )
    camera = dict(
        up=dict(x=0, y=1, z=0)
    )
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Z',  # Flip Y/Z
            zaxis_title='Y',  # Flip Y/Z
            camera=camera,
            dragmode='turntable'
        ),
        scene_camera=camera
    )
    fig.show()


if __name__ == '__main__':
    plot_dragon()
