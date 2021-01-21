import gc
import os
import time
from typing import Optional, Tuple

import numba
import numpy as np

from data.chunks import ChunkGrid
from filters.dilate import dilate
from mathlib import Vec3f, normalize_vec
from render.cloud_render import CloudRender
from render.voxel_render import VoxelRender
from utils import timed


def make_normal_kernel(shape: Tuple[int, int, int] = (3, 3, 3)) -> np.ndarray:
    assert len(shape) == 3
    center = np.asanyarray(shape) // 2
    normals = np.full((*shape, 3), 0, dtype=np.float32)
    for i in np.ndindex(shape):
        normals[i] = i - center
        norm = np.linalg.norm(normals[i])
        if norm > 0:
            normals[i] /= norm
    return normals


def detect_normals(surface: ChunkGrid[np.bool8], outer: ChunkGrid[np.bool8],
                   normal_kernel: Optional[np.ndarray] = None):
    if normal_kernel is None:
        normal_kernel = make_normal_kernel()
    # Method cache
    __np_sum = np.sum

    normal_pos = np.array(list(surface.where()))
    normal_val = np.full((len(normal_pos), 3), 0.0, dtype=np.float32)
    for n, p in enumerate(normal_pos):
        x, y, z = p
        mask: np.ndarray = outer[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        normal_val[n] = __np_sum(normal_kernel[mask], axis=0)
    normal_val = (normal_val.T / np.linalg.norm(normal_val, axis=1)).T
    return normal_pos, normal_val


def grid_normals(surface: ChunkGrid[np.bool8], outer: ChunkGrid[np.bool8], normal_kernel: Optional[np.ndarray] = None) \
        -> ChunkGrid[Vec3f]:
    normal_pos, normal_val = detect_normals(surface, outer, normal_kernel)
    normals: ChunkGrid[np.float32] = ChunkGrid(surface.chunk_size, np.dtype((np.float32, (3,))), 0.0)
    normals[normal_pos] = normal_val
    return normals
