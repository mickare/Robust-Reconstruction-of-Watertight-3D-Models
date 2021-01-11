from typing import Tuple

import numba as nb
import numpy as np
from scipy import ndimage

from mathlib import Vec3i

# @nb.njit(parallel=True)
# def set_array(arr: np.ndarray, pos: np.ndarray, values: np.ndarray):
#     assert len(pos) == len(values)
#     for i in range(len(pos)):
#         arr[pos[i][0], pos[i][1], pos[i][2]] = values[i]
from utils import timed


@nb.njit(parallel=True)
def set_array_1d(arr: np.ndarray, pos: np.ndarray, values: np.ndarray):
    for i in nb.prange(len(pos)):
        arr[pos[i][0], pos[i][1], pos[i][2]] = values[i]


@nb.stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)))
def kernel3(a):
    return 0.14285714285714285 * (a[0, 0, 0]
                                  + a[-1, 0, 0] + a[0, -1, 0] + a[0, 0, -1]
                                  + a[1, 0, 0] + a[0, 1, 0] + a[0, 0, 1])


@nb.njit()
def correlate_iter(shape: Tuple[int, ...], positions, normals: np.ndarray, iterations: int) -> np.ndarray:
    src = np.zeros(shape, dtype=np.float64)
    set_array_1d(src, positions, normals)

    dst = np.zeros(shape, dtype=np.float64)
    for i in range(iterations):
        # _correlate(src, dst, kernel)
        kernel3(src, out=dst)
        # Swap
        src_old = src
        src = dst
        dst = src_old
        # Reset
        set_array_1d(src, positions, normals)
    return src


size = 256
shape = (size, size, size)
count = 1000
positions = np.random.randint(0, size, (count, 3), dtype=np.int64)
positions[0] = (0, 0, 0)
positions[1] = (0, 1, 1)
normals = np.random.random(count).astype(np.float64)

print(positions[:2])
print(normals[:2])

with timed("a: "):
    result = correlate_iter(shape, positions, normals, 10)
    print(result[1, :4, :4])


@nb.jit()
def test():
    return np.zeros(3), np.ones(6)


a, b = test()
print(a, b)
