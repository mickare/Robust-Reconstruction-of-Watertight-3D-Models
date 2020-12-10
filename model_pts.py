from model import ModelLoader
import numpy as np
import os


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
