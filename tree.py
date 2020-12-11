from typing import Optional, Tuple, Dict, Union, Iterator

import numpy as np

from mathlib import Vec3i

ChunkIndex = Tuple[int, int, int]
ChunkSize = 8
ChunkShape = (ChunkSize, ChunkSize, ChunkSize)


class Chunk:
    def __init__(self, index: ChunkIndex):
        self.index = np.array(index, dtype=np.int)
        self.color: Optional[Union[int, np.ndarray]] = None
        self.distance: Union[float, np.ndarray] = 1.0
        self.crust: Union[bool, np.ndarray] = False

    def set_color(self, pos: Vec3i, new_color: int):
        voxel_index = self._inner_index(pos)
        if self.color is None:
            self.color = 0

        if isinstance(self.color, int):
            if new_color != self.color:
                self.color = np.full(ChunkShape, self.color, dtype=np.int)
                self.color[voxel_index] = new_color
            # else nothing to do
        elif isinstance(self.color, np.ndarray):
            self.color[voxel_index] = new_color
        else:
            raise RuntimeError("invalid color type")

    def set_distance(self, pos: Vec3i, distance: float):
        voxel_index = self._inner_index(pos)
        if isinstance(self.distance, float):
            if distance != self.distance:
                self.distance = np.ones(ChunkShape) * self.distance
                self.distance[voxel_index] = distance
            # else nothing to do
        elif isinstance(self.distance, np.ndarray):
            self.distance[voxel_index] = distance
        else:
            raise RuntimeError("invalid distance type")

    def set_crust(self, pos: Vec3i, is_crust: bool):
        voxel_index = self._inner_index(pos)
        if isinstance(self.crust, bool):
            if is_crust != self.crust:
                self.crust = np.ones(ChunkShape) * self.crust
                self.crust[voxel_index] = is_crust
            # else nothing to do
        elif isinstance(self.crust, np.ndarray):
            self.crust[voxel_index] = is_crust
        else:
            raise RuntimeError("invalid distance type")

    def _inner_index(self, pos: Vec3i) -> Tuple[int, int, int]:
        # if not (result < ChunkShape).all():
        #     print("WOW")
        # assert ((0, 0, 0) <= result).all() and (result < ChunkShape).all()
        return tuple(np.asarray(pos, dtype=np.int) % ChunkShape)

    def to_points(self) -> np.ndarray:
        if isinstance(self.distance, float):
            return np.array([]).reshape((0, 3))
        elif isinstance(self.distance, np.ndarray):
            return np.argwhere(self.distance != 1.0) + self.index * ChunkShape
        else:
            raise RuntimeError("invalid distance type")

    def crust_to_points(self) -> np.ndarray:
        if isinstance(self.crust, bool):
            return np.array([]).reshape((0, 3))
        elif isinstance(self.crust, np.ndarray):
            return np.argwhere(self.crust) + self.index * ChunkShape
        else:
            raise RuntimeError("invalid crust type")

    def fill(self, pos: Vec3i, color: int):
        if self.color is None:  # Nicht gefüllt
            self.color = color
        elif isinstance(self.color, int):  # Gefüllt mit einem Wert
            self.color = color
        elif isinstance(self.color, np.ndarray):  # Gefüllt mit unterschiedlichen Werte
            voxel_index = self._inner_index(pos)
            self.color[voxel_index] = color
            # TODO FILL
        else:
            raise RuntimeError("invalid color type!")

    def crust_voxels(self) -> np.ndarray:
        if isinstance(self.crust, bool):
            return np.full(ChunkShape, self.crust, dtype=np.bool)
        elif isinstance(self.crust, np.ndarray):
            return self.crust
        else:
            raise RuntimeError("invalid crust type")


class ChunkGrid:
    def __init__(self, resolution: int = 32):
        assert resolution % ChunkSize == 0
        self.size = resolution // ChunkSize
        self.chunks: Dict[ChunkIndex, Chunk] = dict()

    @property
    def resolution(self) -> int:
        return self.size * ChunkSize

    def __getitem__(self, index) -> Optional[Chunk]:
        assert len(index) == 3
        key = tuple(index)
        return self.chunks.get(key, None)

    def get_chunk_by_index(self, index: Vec3i) -> Optional[Chunk]:
        return self.chunks.get(tuple(index), None)

    def get_chunk(self, pos: Vec3i) -> Optional[Chunk]:
        return self.get_chunk_by_index(self.index(pos))

    def validate_pos(self, pos: Vec3i):
        res = self.resolution
        assert ((0, 0, 0) <= pos).all() and (pos < (res, res, res)).all()

    def validate_index(self, index: Vec3i):
        s = self.size
        assert ((0, 0, 0) <= index).all() and (index < (s, s, s)).all()

    def index(self, pos: Vec3i) -> ChunkIndex:
        indx = np.asarray(pos, dtype=int) // ChunkSize
        self.validate_pos(indx)
        return tuple(indx)

    def index_to_pos(self, index: ChunkIndex) -> Vec3i:
        return np.ndarray(index) * ChunkSize

    def create_if_absent(self, pos: Vec3i) -> Chunk:
        index = self.index(pos)
        c = self.chunks.get(index, None)
        if not c:
            c = Chunk(index)
            self.chunks[index] = c
        return c

    def to_points(self):
        return np.concatenate([c.to_points() for c in self.chunks.values()])

    def crust_to_points(self):
        return np.concatenate([c.crust_to_points() for c in self.chunks.values()])

    def iter_neighbors(self, index: ChunkIndex, flatten=True) -> Iterator[Optional[Chunk]]:
        tmp = np.asarray(index)
        for di in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            i = tmp + di
            c = self.get_chunk_by_index(i)
            if flatten:
                if c:
                    yield c
            else:
                yield c


if __name__ == '__main__':
    from model_pts import PtsModelLoader
    from cloud_render import CloudRender

    data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)

    g = ChunkGrid(16)
    scaled = (data - data_min) / np.max(data_max - data_min) * g.resolution

    for p in scaled:
        pos = np.array(p, dtype=int)
        c = g.create_if_absent(pos)
        c.set_distance(pos, 0)
        c.set_crust(pos, True)

    assert scaled.shape[1] == 3
    pts = g.crust_to_points() + 0.5
    assert pts.shape[1] == 3
    CloudRender().plot(scaled, pts, size=1)
