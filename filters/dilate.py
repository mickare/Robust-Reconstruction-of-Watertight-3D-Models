from typing import Optional, Union

import numpy as np
from scipy import ndimage

from data.chunks import ChunkGrid, Chunk

bool_t = Union[bool, np.bool8]


def dilate_no_mask(image: ChunkGrid[bool_t], structure: Optional[np.ndarray] = None, steps=1) -> ChunkGrid[bool_t]:
    if structure is not None:
        assert structure.ndim == 2 and structure.shape == (3, 3)

    # Method cache (prevent lookup in loop)
    __ndimage_binary_dilation = ndimage.binary_dilation
    __grid_ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index
    __chunk_padding = Chunk.padding
    __chunk_set_array = Chunk.set_array

    result = image
    for step in range(steps):
        # Temporary result between each step
        tmp = result.copy(empty=True)
        # Dilate inner chunk
        for r in result.chunks:
            dil = __ndimage_binary_dilation(__chunk_padding(r, result, 1), structure=structure)
            c = __grid_ensure_chunk_at_index(tmp, r.index)
            __chunk_set_array(c, dil[1:-1, 1:-1, 1:-1])
        # Set result
        result = tmp
    result.cleanup()
    return result


def dilate(image: ChunkGrid[bool_t], structure: Optional[np.ndarray] = None, mask: ChunkGrid[bool] = None, steps=1) \
        -> ChunkGrid[bool_t]:
    if mask is None:
        return dilate_no_mask(image, structure, steps)
    else:
        raise NotImplementedError
