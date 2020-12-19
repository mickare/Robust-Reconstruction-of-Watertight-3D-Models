from typing import Optional, Tuple, Union, Iterator

import numpy as np
from scipy.ndimage import binary_dilation

from data.index_dict import IndexDict
from mathlib import Vec3i

ChunkIndex = Tuple[int, int, int]
ChunkSize = 8
ChunkShape = (ChunkSize, ChunkSize, ChunkSize)


class Chunk:
    __slots__ = ['index', 'color', 'distance', 'crust']

    def __init__(self, index: ChunkIndex):
        self.index = np.array(index, dtype=np.int)
        self.color: Union[int, np.ndarray] = 0
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

    def color_to_points(self, color) -> np.ndarray:
        if isinstance(self.color, int):
            return np.array([]).reshape((0, 3))
        elif isinstance(self.crust, np.ndarray):
            return np.argwhere(np.array([[[True if x == color else False for x in y] for y in z] for z in self.color])) + self.index * ChunkShape
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

    def color_voxels(self) -> np.ndarray:
        if isinstance(self.color, np.ndarray):
            return self.color
        elif isinstance(self.color, int):
            return np.full(ChunkShape, self.color, dtype=int)
        else:
            raise RuntimeError("invalid color type")




class ChunkData:
    def __init__(self, resolution: int = 32):
        assert resolution % ChunkSize == 0
        self.size = resolution // ChunkSize
        self.chunks: IndexDict[Chunk] = IndexDict()

    @property
    def resolution(self) -> int:
        return self.size * ChunkSize

    def chunk(self, index: Vec3i) -> Optional[Chunk]:
        return self.chunks.get(tuple(index), None)

    def pos(self, pos: Vec3i) -> Optional[Chunk]:
        return self.chunk(self.index(pos))

    def index(self, pos: Vec3i) -> ChunkIndex:
        indx = np.asarray(pos, dtype=int) // ChunkSize

        return tuple(indx)

    def index_to_pos(self, index: ChunkIndex) -> Vec3i:
        return np.array(index) * ChunkSize

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

    def color_to_points(self):
        color = 1
        points = []
        while True:
            colors = np.concatenate([c.color_to_points(color) for c in self.chunks.values()])
            if not colors.any():
                break
            points.append(colors)
            color += 1
        return np.array(points)

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


def dilate(grid: ChunkData):
    def set_crust(index_chunk: Vec3i, index_neighbor: Vec3i, axis: int, crust_segment: np.ndarray, index_crust: int):
        def get_index(i, j):
            a = [i, j]
            a.insert(axis, index_crust)
            return tuple(a)

        index = np.asarray(index_chunk) + index_neighbor
        if not all(i >= 0 for i in index):
            return
        c = grid.get_chunk_by_index(index)
        if not c:
            c = g.create_if_absent(g.index_to_pos(index))
        c_crust = c.crust_voxels()
        for i, row in enumerate(crust_segment):
            for j, e in enumerate(row):
                if e:
                    c_crust[get_index(i, j)] = True
        c.crust = c_crust

    for chunk in grid.chunks:
        crust = grid.get_chunk_by_index(chunk).crust
        if isinstance(crust, np.ndarray):
            grid.get_chunk_by_index(chunk).crust = binary_dilation(crust)
    grid_chunks = grid.chunks.copy()
    for chunk in grid_chunks:
        crust = grid.get_chunk_by_index(chunk).crust
        if crust is False:
            continue
        indices = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        if isinstance(crust, bool) and crust is True:
            i, coord = 0, 2
            ind, coords = [-1, 0], [1, 2, 0]
            for j, c in enumerate(grid.iter_neighbors(chunk, flatten=False)):
                if i == 0:
                    coord = coords[coord]
                i = ind[i]
                index = np.asarray(chunk) + indices[j]
                if not all(i >= 0 for i in index):
                    continue
                if not c:
                    c = g.create_if_absent(g.index_to_pos(np.asarray(chunk) + indices[j]))
                c_crust = c.crust_voxels()
                np.put_along_axis(c_crust, np.array([[[i]]]), 1, coord)
        elif isinstance(crust, np.ndarray):
            dim = range(3)
            ind = [-1, 0]

            i_iter = iter(indices)
            for d in dim:
                for i in ind:
                    if crust.take(ind[i], d).any():
                        set_crust(chunk, next(i_iter), d, crust.take(ind[i], d), i)
                    else:
                        next(i_iter)


def flood_fill(grid: ChunkGrid):
    for ind in grid.chunks:
        chunk = grid.get_chunk_by_index(ind)
        chunk.color = 0
    color = 1
    while flood_fill_recursive(grid, color):
        print(color)
        color += 1


def flood_fill_recursive(grid: ChunkGrid, color: int):
    current_chunk = None
    for chunk in grid.chunks:
        crust = grid.get_chunk_by_index(chunk).crust
        if not isinstance(crust, np.ndarray):
            if not crust and not grid.get_chunk_by_index(chunk).color:
                current_chunk = chunk
        else:
            c_color = grid.get_chunk_by_index(chunk).color_voxels()
            print(len(np.argwhere(np.logical_not(crust))))
            print(len(np.argwhere(np.logical_not(c_color))))
            mask = np.argwhere(np.logical_and(np.logical_not(c_color), np.logical_not(crust)))
            print(len(mask))
            if len(mask) > 0:
                grid.get_chunk_by_index(chunk).color = c_color
                c_color[mask[0]] = color
                current_chunk = chunk
        if current_chunk:
            break
    if not current_chunk:
        return False
    chunk_queue = []
    indices = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    while True:
        new_fill = flood_fill_chunk(grid, current_chunk, color)
        if new_fill:
            for i in indices:
                neighbor = np.asarray(current_chunk) + i
                if not all(0 <= j < grid.resolution for j in neighbor):
                    continue
                if not grid.get_chunk_by_index(neighbor):
                    grid.create_if_absent(g.index_to_pos(neighbor))
                if isinstance(grid.get_chunk_by_index(neighbor).crust, np.ndarray):
                    flood_fill_border(grid, current_chunk, neighbor, color)
                chunk_queue.append(neighbor)
        if chunk_queue:
            current_chunk = chunk_queue.pop()
        else:
            break
    return True


def flood_fill_chunk(grid: ChunkGrid, chunk_index: Vec3i, color: int):
    chunk = grid.get_chunk_by_index(chunk_index)
    if not isinstance(chunk.crust, np.ndarray) or not chunk.crust.any():
        if not chunk.color:
            chunk.color = color
            return True
        else:
            return False
    else:
        chunk_color = chunk.color_voxels()
        mask = np.logical_and(np.logical_not(chunk_color), np.logical_not(chunk.crust))
        filling = np.array([[[True if x == color else False for x in y] for y in z] for z in chunk_color])
        filling = binary_dilation(filling, iterations=-1, mask=mask)
        filling = np.where(filling, np.full(shape=chunk_color.shape, fill_value=color), chunk_color)
        if np.array_equal(filling, chunk_color):
            return False
        chunk.color = filling
        return True


def flood_fill_border(grid: ChunkGrid, source_chunk_index: Vec3i, target_chunk_index: Vec3i, color: int):
    def copy_border(target_index: int):
        def get_index(i, j):
            a = [i, j]
            a.insert(axis, target_index)
            return tuple(a)

        c = grid.get_chunk_by_index(target_chunk_index)
        if not c:
            c = g.create_if_absent(g.index_to_pos(target_chunk_index))
        c_color = c.color_voxels()
        for i, row in enumerate(segment):
            for j, e in enumerate(row):
                if e == color:
                    index = get_index(i, j)
                    if not c_color[index]:
                        c_color[index] = color
        c.color = c_color

    source_chunk = grid.get_chunk_by_index(source_chunk_index)
    segment = None if isinstance(source_chunk.color, np.ndarray) else np.full(ChunkShape[1:], dtype=int,
                                                                              fill_value=color)

    axis = -1
    for i in range(3):
        if source_chunk_index[i] != target_chunk_index[i]:
            axis = i
            break
    if axis == -1:
        raise RuntimeError("Indices are equal.")

    if source_chunk_index[axis] < target_chunk_index[axis]:
        segment = segment if isinstance(segment, np.ndarray) else source_chunk.color.take(0, axis)
        copy_border(-1)
    elif source_chunk_index[axis] > target_chunk_index[axis]:
        segment = segment if isinstance(segment, np.ndarray) else source_chunk.color.take(-1, axis)
        copy_border(0)
    else:
        raise RuntimeError("Invalid index.")


if __name__ == '__main__':
    from model.model_pts import PtsModelLoader
    from render_cloud import CloudRender

    data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)

    g = ChunkData(16)
    scaled = (data - data_min) / np.max(data_max - data_min) * g.resolution

    for p in scaled:
        pos = np.array(p, dtype=int)
        c = g.create_if_absent(pos)
        c.set_distance(pos, 0)
        c.set_crust(pos, True)

    assert scaled.shape[1] == 3
    pts = g.crust_to_points() + 0.5
    assert pts.shape[1] == 3
    fig = CloudRender().plot(scaled, pts, size=1)
    fig.show()

    # Dilation
    dilate(g)
    pts = g.crust_to_points() + 0.5
    fig = CloudRender().plot(scaled, pts, size=1)
    fig.show()

    # Flood filling
    flood_fill(g)
    pts = g.color_to_points() + 0.5
    args = [scaled]
    args.extend([i for i in pts])
    args = (i for i in args)
    fig = CloudRender().plot(*args, size=1)
    fig.show()
