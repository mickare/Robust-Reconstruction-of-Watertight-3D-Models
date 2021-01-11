from typing import TypeVar

import numpy as np
from scipy import ndimage

from data.chunks import Chunk, ChunkGrid

V = TypeVar('V')
M = TypeVar('M')




def correlate_grid(grid: ChunkGrid, kernel: np.ndarray, iterations=1) -> "ChunkGrid[M]":
    return _correlate_filter(grid, kernel, iterations, convolve=False)


def convolve_grid(grid: ChunkGrid, kernel: np.ndarray, iterations=1) -> "ChunkGrid[M]":
    return _correlate_filter(grid, kernel, iterations, convolve=True)
