from typing import Optional, Tuple, Union, Iterator

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation

from data.chunks import ChunkGrid, Chunk, ChunkIndex, ChunkFace
from mathlib import Vec3i


def dilate_no_mask(image: ChunkGrid[bool], structure: Optional[np.ndarray] = None, steps=1) -> ChunkGrid[bool]:
    if structure is not None:
        assert structure.ndim == 2 and structure.shape == (3, 3)

    result = image
    for step in range(steps):
        # Temporary result between each step
        tmp = result.copy(empty=True)
        # Dilate inner chunk
        for r in result.chunks:
            dil = ndimage.binary_dilation(r.padding(result, 1), structure=structure)
            c = tmp.ensure_chunk_at_index(r.index)
            c.set_array(dil[1:-1, 1:-1, 1:-1])
        # Set result
        result = tmp
    result.cleanup()
    return result


def dilate(image: ChunkGrid[bool], structure: Optional[np.ndarray] = None, mask: ChunkGrid[bool] = None, steps=1) \
        -> ChunkGrid[bool]:
    if mask is None:
        return dilate_no_mask(image, structure, steps)
    else:
        raise NotImplementedError
