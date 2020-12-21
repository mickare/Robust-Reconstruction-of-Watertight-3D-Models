from typing import Optional, Tuple, Union, Iterator

import numpy as np
from scipy.ndimage import binary_dilation

from data.chunks import ChunkGrid, Chunk, ChunkIndex, ChunkFace, ChunkType
from mathlib import Vec3i


def dilate(grid: ChunkGrid[int]):
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
        if chunk.type == ChunkType.ARRAY:
            chunk.set_array(binary_dilation(chunk.value))


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
