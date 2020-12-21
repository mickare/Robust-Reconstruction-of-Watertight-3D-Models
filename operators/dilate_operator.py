from typing import Optional, Tuple, Union, Iterator

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation

from data.chunks import ChunkGrid, Chunk, ChunkIndex, ChunkFace
from mathlib import Vec3i


def dilate(image: ChunkGrid[bool], mask: ChunkGrid[bool] = None) -> ChunkGrid[bool]:
    if mask is None:
        mask = ChunkGrid(image.chunk_size, dtype=bool, fill_value=True)
    else:
        mask = mask.astype(bool)

    result = image.copy()

    # Dilate inner chunk
    for r in result.chunks:
        tmp = ndimage.binary_dilation(r.to_array(), mask=mask.ensure_chunk_at_index(r.index, insert=False).to_array())
        r.set_array(tmp)

    # Dilate chunk overflow
    for index in list(result.chunks.keys()):
        for f, n in result.iter_neighbors_indicies(index):
            r_n = result.ensure_chunk_at_index(n)
            m = mask.ensure_chunk_at_index(index, insert=False).to_array()
            img = image.ensure_chunk_at_index(index, insert=False).to_array()
            s0 = f.slice()
            s1 = f.flip().slice()
            r_n[s1] |= m[s0] & img[s0]

            r_n.cleanup_memory()
    return result
