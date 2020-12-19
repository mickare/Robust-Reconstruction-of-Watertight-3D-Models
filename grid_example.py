from render_cloud import render_cloud
from model.model_pts import PtsModelLoader

import numpy as np

data = PtsModelLoader().load("models/bunny/bunnyData.pts")

resolution = 128


class Grid:
    def __init__(self, points: np.ndarray, resolution: int):
        self.grid = np.full((resolution, resolution, resolution), 1, dtype=float)
        self.boundaries = []
        for i in range(3):
            diff = self._find_max(points, i) - self._find_min(points, i)
            self.boundaries.append([self._find_min(points, i) - diff, self._find_max(points, i) + diff])
        self.steps = [(self.boundaries[i][1] - self.boundaries[i][0]) / resolution for i in range(3)]
        self.resolution = resolution
        self._init_confidence(points)

    def _init_confidence(self, points: np.ndarray):
        for p in points:
            self.grid[int((p[0] + abs(self.boundaries[0][0])) / self.steps[0])][
                int((p[1] + abs(self.boundaries[1][0])) / self.steps[1])][
                int((p[2] + abs(self.boundaries[2][0])) / self.steps[2])] = 0

    def to_points(self):
        points = [[self.boundaries[i][0] + idx[i] * self.steps[i] for i in range(3)] for idx, x in
                  np.ndenumerate(self.grid) if x < 1]
        return np.array(points, dtype=float)

    def dilate(self):
        for idx, x in np.ndenumerate(self.grid):
            if x < 1:
                pass
                #self.add_to_crust(idx, x)

    def _add_to_crust(self):
        pass

    @staticmethod
    def _find_min(array: np.ndarray, coord: int) -> float:
        return np.amin(array.T[coord])

    @staticmethod
    def _find_max(array: np.ndarray, coord: int) -> float:
        return np.amax(array.T[coord])


grid = Grid(data, resolution)
points = grid.to_points()

render_cloud(grid.to_points())
