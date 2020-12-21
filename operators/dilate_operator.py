from typing import Optional, Tuple, Union, Iterator

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation

from data.chunks import ChunkGrid, Chunk, ChunkIndex, ChunkFace
from mathlib import Vec3i


def dilate(image: ChunkGrid[bool], mask: ChunkGrid[bool] = None) -> ChunkGrid[bool]:
    if mask is None:
        mask = ChunkGrid(image.chunk_size, dtype=bool, empty_value=True)
    else:
        mask = mask.astype(bool)

    result = image.copy()

    # Dilate inner chunk
    for r in result.chunks:
        tmp = ndimage.binary_dilation(r.to_array(), mask=mask.ensure_chunk_at_index(r.index, insert=False).to_array())
        # Dilate chunk overflow
        for f, c in image.iter_neighbors(r.index, flatten=True):
            tmp[f.slice()] = c.to_array()[f.flip().slice()]

        r.set_array(tmp)
        r.cleanup_memory()

    return result
