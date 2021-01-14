import queue
from typing import Optional, Union

import numpy as np
from scipy import ndimage

from data.chunks import ChunkGrid, Chunk
from data.faces import ChunkFace

bool_t = Union[bool, np.bool8]


def dilate(image: ChunkGrid[bool_t], steps=1, structure: Optional[np.ndarray] = None, mask: ChunkGrid[bool] = None) \
        -> ChunkGrid[bool_t]:
    if mask is None:
        return dilate_no_mask(image, steps, structure)
        # if steps < image.chunk_size:
        #     return dilate_no_mask(image, structure, steps)
        # else:
        #     return dilate_no_mask_fast(image, structure, steps)
    else:
        raise NotImplementedError


def dilate_no_mask(image: ChunkGrid[bool_t], steps=1, structure: Optional[np.ndarray] = None) -> ChunkGrid[bool_t]:
    if structure is not None:
        assert structure.ndim == 2 and structure.shape == (3, 3)

    __pad_slice = slice(1, -1)

    result = image.astype(np.bool8)
    for step in range(steps):
        # Temporary result between each step
        tmp = result.copy(empty=True)
        # Dilate inner chunk
        # result.pad_chunks(1)

        for index, r in result.chunks.items():
            if r.is_filled() and r.value:
                tmp.ensure_chunk_at_index(index).set_fill(r.value)
                for f, ni in ChunkGrid.iter_neighbors_indices(r.index):
                    ch = tmp.ensure_chunk_at_index(ni)
                    if not (ch.is_filled() and ch.value):
                        arr = ch.to_array()
                        arr[f.flip().slice()] = True
                        ch.set_array(arr)
                        ch.cleanup()
                continue

            padded = result.padding_at(index, 1, corners=False, edges=False)
            if (not np.any(padded)):  # Skip, nothing to do
                continue

            # Do dilation
            dilated = ndimage.binary_dilation(padded, structure=structure)

            # Copy result to tmp
            ch = tmp.ensure_chunk_at_index(index)
            ch.set_array(dilated[1:-1, 1:-1, 1:-1])
            ch.cleanup()

            # Propagate to the next chunks
            for f in ChunkFace:  # type: ChunkFace
                s = dilated[f.slice(other=__pad_slice)]
                if np.any(s):
                    ch: Chunk = tmp.ensure_chunk_at_index(f.direction() + index)
                    arr = ch.to_array()
                    arr[f.flip().slice()] |= s
                    ch.set_array(arr)

        # Set result
        result = tmp
    result.cleanup(remove=True)
    return result

# def dilate_no_mask_fast(image: ChunkGrid[bool_t], structure: Optional[np.ndarray] = None, steps=1) -> ChunkGrid[bool_t]:
#     if structure is not None:
#         assert structure.ndim == 2 and structure.shape == (3, 3)
#
#     # Only allow dilation on not fully filled spaces
#     assert not image.fill_value
#
#     # Method cache (prevent lookup in loop)
#     __ndimage_binary_dilation = ndimage.binary_dilation
#     __grid_ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index
#     __chunk_set_array = Chunk.set_array
#
#     result = image
#     size = image.chunk_size
#
#     remaining = steps
#     step = 0
#     while remaining > 0:
#         # Binary dilation iterations in this step
#         iterations = min(remaining, image.chunk_size)
#         if iterations <= 0:
#             break
#
#         # Temporary result between each major step
#         tmp = result.copy(empty=True)
#
#         # Dilate inner chunk
#         result.pad_chunks(1)
#         for r in result.chunks:
#             if r.is_filled() and r.value:  # Skip, nothing to do
#                 continue
#
#             src = result.get_block_at(r.index, (3, 3, 3), edges=True, corners=True)
#             if all(b.is_filled() and not b.value for b in np.flatiter(src)):  # Skip, nothing to do
#                 continue
#
#             # Do dilation
#             padded = result.block_to_array(src)
#             dil = __ndimage_binary_dilation(padded, structure=structure, iterations=iterations)
#
#             # Copy result to tmp
#             tmp.set_block_at(r.index, dil, replace=False)
#             tmp.ensure_chunk_at_index(r.index).set_array(dil[size:size + size, size:size + size, size:size + size])
#
#         # Set result
#         result = tmp
#         result.cleanup(remove=True)
#         step += 1
#
#     return result
