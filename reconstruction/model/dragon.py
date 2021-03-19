import os
from typing import Tuple, Iterator, Optional

import numpy as np

from .model_ply import PlyModelLoader

dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = os.path.join(dir_path, "../../models")


class Dragon:

    @classmethod
    def load(cls, path: Optional[str] = None):
        if path is None:
            path = os.path.join(default_path, "../../models/dragon_recon/dragon_vrip.ply")
        return PlyModelLoader().load(path)


class MergedDragon:
    """
    Deprecated - trying to load all partial Stanford Dragon scans.
    NOT WORKING! Use the dragon loader above.
    """
    conf = [
        "dragon_backdrop/carvers.conf",
        "dragon_fillers/fillers.conf",
        "dragon_side/dragonSideRight.conf",
        "dragon_stand/dragonStandRight.conf",
        "dragon_up/dragonUpRight.conf"
    ]

    def __init__(self, path: str = default_path):
        self.path = path

    def load_models(self) -> Iterator[Tuple[str, np.ndarray]]:
        loader = PlyModelLoader()
        for c in self.conf:
            yield c, loader.load(os.path.join(self.path, c))

    def load(self):
        models = [m for c, m in self.load_models()]
        return np.concatenate(models)


def plot_dragon():
    data = Dragon.load()

    from reconstruction.render.cloud_render import CloudRender
    ren = CloudRender()
    fig = ren.make_figure()
    fig.add_trace(ren.make_scatter(data, size=1))
    fig.show()


if __name__ == '__main__':
    plot_dragon()
