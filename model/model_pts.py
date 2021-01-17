import os
from typing import Tuple, Sequence

import numpy as np

from model.loader import ModelLoader


class PtsModelLoader(ModelLoader):
    def load(self, path: str) -> np.ndarray:
        assert os.path.isfile(path)
        with open(path, mode='tr') as fp:
            file = fp.readlines()
            coords = np.zeros((len(file), 3), dtype=float)
            num_coords = 0
            for l in file:
                coord = l.split(" ")
                if len(coord) != 3:
                    continue
                coords[num_coords] = coord
                num_coords += 1
            coords = np.delete(coords, slice(num_coords, coords.shape[0]), 0)
        return coords


class FixedPtsModels:

    @classmethod
    def _bunny_fix_baseplate(cls, model: np.ndarray, res=0.002) -> Sequence[np.ndarray]:
        ymin = np.min(model.T[1])

        base0 = FixedPtsModels.yplate(ymin, (-0.05, -0.01), (0.02, 0.04), res=res)
        base1 = FixedPtsModels.yplate(ymin, (0.023, 0.001), (0.035, 0.025), res=res)
        base2 = FixedPtsModels.yplate(ymin, (0.012, -0.022), (-0.04, -0.011), res=res)
        base3 = [
            (-0.026, ymin, 0.045),
            (-0.018, ymin, 0.045),
            (-0.011, ymin, 0.045),
        ]
        return base0, base1, base2, base3

    @classmethod
    def bunny(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(dir_path, "../models/bunny/bunnyData.pts")
        model = PtsModelLoader().load(file)

        base = np.concatenate(cls._bunny_fix_baseplate(model))
        return np.concatenate((model, base))

    @classmethod
    def yplate(cls, y: float, p0: Tuple[float, float], p1: Tuple[float, float], res=0.1):
        pmin = np.min((p0, p1), axis=0)
        pmax = np.max((p0, p1), axis=0)
        ns = np.ceil((pmax - pmin) // res).astype(int)
        xs = np.linspace(pmin[0], pmax[0], max(1, ns[0]))
        zs = np.linspace(pmin[1], pmax[1], max(1, ns[1]))
        xv, zv = np.meshgrid(xs, zs)
        return [(x, y, z) for x, z in zip(xv.flatten(), zv.flatten())]


if __name__ == '__main__':
    from render.cloud_render import CloudRender

    data = FixedPtsModels.bunny()

    ren = CloudRender()
    fig = ren.make_figure()
    fig.add_trace(ren.make_scatter(data))
    fig.show()
