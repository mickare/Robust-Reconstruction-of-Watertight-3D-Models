from typing import Optional, Union

import numpy as np
from scipy import ndimage

from data.chunks import ChunkGrid, Chunk
from data.faces import ChunkFace

bool_t = Union[bool, np.bool8]


def dilate_no_mask(image: ChunkGrid[bool_t], structure: Optional[np.ndarray] = None, steps=1) -> ChunkGrid[bool_t]:
    if structure is not None:
        assert structure.ndim == 2 and structure.shape == (3, 3)

    # Method cache (prevent lookup in loop)
    __ndimage_binary_dilation = ndimage.binary_dilation
    __grid_ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index
    __chunk_padding = Chunk.padding
    __chunk_set_array = Chunk.set_array

    __pad_slice = slice(1, -1)

    result = image
    for step in range(steps):
        # Temporary result between each step
        tmp = result.copy(empty=True)
        # Dilate inner chunk
        # result.pad_chunks(1)
        for r in result.chunks:
            if r.is_filled() and r.value:  # Skip, nothing to do
                continue

            padded = __chunk_padding(r, result, 1)
            if not np.any(padded):  # Skip, nothing to do
                continue

            # Do dilation
            dil = __ndimage_binary_dilation(padded, structure=structure)

            # Copy result to tmp
            c = __grid_ensure_chunk_at_index(tmp, r.index)
            __chunk_set_array(c, dil[1:-1, 1:-1, 1:-1])

            # Propagate to the next chunks
            for f in ChunkFace:  # type: ChunkFace
                s = dil[f.slice(other=__pad_slice)]
                if np.any(s):
                    neighbor: Chunk = __grid_ensure_chunk_at_index(tmp, f.direction() + r.index)
                    arr = neighbor.to_array()
                    arr[f.flip().slice()] |= s
                    neighbor.set_array(arr)

        # Set result
        result = tmp
    result.cleanup(remove=True)
    return result


def dilate_no_mask_fast(image: ChunkGrid[bool_t], structure: Optional[np.ndarray] = None, steps=1) -> ChunkGrid[bool_t]:
    if structure is not None:
        assert structure.ndim == 2 and structure.shape == (3, 3)

    # Method cache (prevent lookup in loop)
    __ndimage_binary_dilation = ndimage.binary_dilation
    __grid_ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index
    __chunk_padding = Chunk.padding
    __chunk_set_array = Chunk.set_array

    result = image
    size = image.chunk_size

    remaining = steps
    step = 0
    while remaining > 0:
        # Binary dilation iterations in this step
        iterations = min(remaining, image.chunk_size)
        if iterations <= 0:
            break

        # Temporary result between each major step
        tmp = result.copy(empty=True)

        # Dilate inner chunk
        result.pad_chunks(1)
        for r in result.chunks:
            if r.is_filled() and r.value:  # Skip, nothing to do
                continue

            src = result.get_block_at(r.index, (3, 3, 3), ensure=True, insert=False)
            if all(b.is_filled() and not b.value for b in np.flatiter(src)):  # Skip, nothing to do
                continue

            # Do dilation
            padded = Chunk.block_to_array(src)
            dil = __ndimage_binary_dilation(padded, structure=structure, iterations=iterations)

            # Copy result to tmp
            tmp.set_block_at(r.index, dil, replace=False)
            tmp.ensure_chunk_at_index(r.index).set_array(dil[size:size + size, size:size + size, size:size + size])

        # Set result
        result = tmp
        result.cleanup(remove=True)
        step += 1

    return result


def dilate(image: ChunkGrid[bool_t], structure: Optional[np.ndarray] = None, steps=1, mask: ChunkGrid[bool] = None) \
        -> ChunkGrid[bool_t]:
    if mask is None:
        if steps < image.chunk_size:
            return dilate_no_mask(image, structure, steps)
        else:
            return dilate_no_mask_fast(image, structure, steps)
    else:
        raise NotImplementedError
