import hashlib
import os
from dataclasses import dataclass
from typing import Optional, List, Sequence, Iterator

import numpy as np
import plyfile
import plotly.graph_objects as go
import tqdm

from mathlib import quaternion_rotation_matrix
from model.model import ModelLoader


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
    def read_conf_file(cls, path: str) -> str:
        with open(path, mode='tr') as fp:
            return fp.readlines()

    @classmethod
    def load_ply_conf(cls, path: str):
        assert os.path.isfile(path)

        base = os.path.dirname(path)
        conf = cls.read_conf_file(path)

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
        for scan in tqdm.tqdm(self.scans, desc="Loading ply points"):
            yield scan.points()


class PlyModelLoader(ModelLoader):

    def load_file(self, path: str) -> np.ndarray:
        folder = ScanFolder.load_ply_conf(path)
        pbar = tqdm.tqdm(folder.scans, desc=f"Loading {path}")
        pts = []
        for s in pbar:  # type: Scan
            fname = os.path.basename(s.file)
            pbar.set_description(f"Loading {fname}")
            pts.append(s.points())

        return folder.transform.apply(np.concatenate(pts))

    def load_cache(self, path: str):
        fname = os.path.basename(path)
        conf = ScanFolder.read_conf_file(path)
        h = hashlib.sha256()
        h.update(path.encode('utf-8'))
        h.update("\n".join(conf).encode('utf-8'))
        key = h.hexdigest()

        file = os.path.join("../.cache", f"{key}_ply.npy")
        if os.path.isfile(file):
            print(f"Loading {fname} from Cache...")
            data = np.load(file, allow_pickle=False)
        else:
            print(f"Loading {fname} from File...")
            data = self.load_file(path)
            os.makedirs("../.cache")
            np.save(file, data, allow_pickle=False)

        return data

    def load(self, path: str) -> np.ndarray:
        return self.load_cache(path)


def plot_dragon():
    import plotly.graph_objects as go
    import tqdm

    scans = ScanFolder.load_ply_conf("../models/dragon_stand/dragonStandRight.conf")

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
