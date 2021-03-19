"""
Approximation of the medial axis in a voxel model by propagating normals.

The general idea is described in the paper.

It estimates the normals on the outer crust and then propagates normals into voxels that are not yet occupied.
The normal field then grows inwards the model.
"""

from typing import Optional, Tuple, Dict

import numba
import numpy as np
from scipy import ndimage
import plotly.graph_objects as go

from reconstruction.data.chunks import ChunkGrid
from reconstruction.filters.dilate import dilate
from reconstruction.mathlib import Vec3f, normalize_vec
from reconstruction.render.cloud_render import CloudRender
from reconstruction.render.voxel_render import VoxelRender
from reconstruction.utils import timed

_CONST_NORMAL_DIRECTIONS = np.array([
    normalize_vec(np.array(p, dtype=np.float32) - 1) if p != (1, 1, 1) else (0, 0, 0) for p in np.ndindex(3, 3, 3)
], dtype=np.float32)


@numba.njit(parallel=True, fastmath=True)
def normal_cone_angles(normals: np.ndarray, mask: np.ndarray, threshold=0.5 * np.pi, min_norm: float = 1e-15):
    assert normals.ndim == 4
    size = normals.shape[0]
    assert normals.shape == (size, size, size, 3)
    assert mask.shape == (size, size, size)

    result = np.zeros((size - 2, size - 2, size - 2), dtype=np.bool8)
    for i in numba.pndindex((size - 2, size - 2, size - 2)):
        # Collect normals for position i
        current = np.empty((26, 3), dtype=np.float32)  # 26 possible neighbors
        ci: numba.uint32 = 0
        for n_o, o in enumerate(np.ndindex((3, 3, 3))):
            if o != (1, 1, 1):
                x, y, z = i[0] + o[0], i[1] + o[1], i[2] + o[2]
                if mask[x, y, z]:
                    value = normals[x, y, z]
                    norm = np.linalg.norm(value)
                    if norm > min_norm:  # Only add if norm is valid
                        current[ci] = value / norm
                        ci += 1
        if ci > 3:
            valid = current[:ci]
            # Check angle between all valid normals
            result[i[0], i[1], i[2]] = np.any(np.arccos(valid @ valid.T) > threshold)
    return result


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


@numba.njit(parallel=True)
def set_array_1d(arr: np.ndarray, pos: np.ndarray, values: np.ndarray):
    for i in numba.prange(len(pos)):
        arr[pos[i][0], pos[i][1], pos[i][2]] = values[i]


@numba.stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)), cval=False)
def kernel3(a):
    return numba.boolean((a[0, 0, 0]
                          | a[-1, 0, 0] | a[0, -1, 0] | a[0, 0, -1]
                          | a[1, 0, 0] | a[0, 1, 0] | a[0, 0, 1]))


@numba.njit(fastmath=True, parallel=True)
def _collect_normals_at(normals: np.ndarray, mask: np.ndarray, positions: np.ndarray):
    assert positions.ndim == 2 and positions.shape[1] == 3
    pos_ignore = normals.shape[0] - 1
    for pi in numba.prange(len(positions)):
        pos = positions[pi]
        # Ignore bounding
        if np.any(pos == 0) or np.any(pos == pos_ignore):
            continue

        # Propagate only normals that are already set in mask_prev
        count = 0
        sum = np.zeros(3, np.float32)
        for off in np.ndindex(3, 3, 3):
            poff = pos + np.asarray(off) - 1
            if mask[poff[0], poff[1], poff[2]]:
                sum += normals[poff[0], poff[1], poff[2]]
                count += 1
        if count > 0:
            vec = sum / count
            normals[pos[0], pos[1], pos[2]] = vec / np.linalg.norm(vec)


@numba.njit(fastmath=True, parallel=True)
def _block_propagate_normals(normals: np.ndarray, mask: np.ndarray, max_iterations: int = -1) \
        -> Tuple[np.ndarray, np.ndarray]:
    size3 = normals.shape[0]
    size = size3 // 3
    assert normals.shape == (size3, size3, size3, 3)
    assert mask.shape == (size3, size3, size3)
    mask_prev = mask

    if max_iterations < 0:
        max_iterations = size
    if np.any(mask):
        mask_next = np.empty_like(mask)

        for i in range(min(size, max_iterations)):
            # Standard kernel on mask to detect where to propagate a normal next
            mask_next = kernel3(mask_prev, out=mask_next)
            changed = np.argwhere(mask_prev ^ mask_next)

            if len(changed) == 0:
                break
            _collect_normals_at(normals, mask_prev, changed)

            # Swap
            mask_prev_old = mask_prev
            mask_prev = mask_next
            mask_next = mask_prev_old
    return normals, mask_prev


def propagate_normals(iterations: int, values: ChunkGrid[Vec3f], positions: ChunkGrid[np.bool8],
                      mask: ChunkGrid[np.bool8]) -> Tuple[ChunkGrid[np.float32], ChunkGrid[np.bool8]]:
    assert iterations >= 0

    values = values.copy()
    positions = positions.copy()
    positions.cleanup(remove=True)

    # Find indices where to operate
    indices_offset = positions.chunks.minmax()[0]
    indices = [tuple(i) for i in np.array(list(np.ndindex(*positions.chunks.size())), dtype=np.int) + indices_offset]
    indices = set(i for i in indices if mask.ensure_chunk_at_index(i, insert=False).any())

    for i in range(iterations):
        tmp_values = values.copy(empty=True)
        tmp_positions = positions.copy(empty=True)

        count_changed = 0
        for index in positions.chunks.keys():
            if index not in indices:
                continue
            pad_mask = positions.ensure_chunk_at_index(index, insert=False).padding(positions, 1, corners=True)
            if not np.any(pad_mask):
                continue
            pad_normals = values.ensure_chunk_at_index(index, insert=False).padding(values, 1, corners=True)

            dil_mask = ndimage.binary_dilation(pad_mask)
            changed = pad_mask ^ dil_mask

            if np.any(changed):
                _collect_normals_at(pad_normals, pad_mask, np.argwhere(changed))
                ch_result = tmp_values.ensure_chunk_at_index(index)
                ch_mask = tmp_positions.ensure_chunk_at_index(index)
                ch_result.set_array(pad_normals[1:-1, 1:-1, 1:-1])
                ch_mask.set_array(dil_mask[1:-1, 1:-1, 1:-1])
                count_changed += 1

        values = tmp_values
        positions = tmp_positions

        if count_changed == 0:  # Nothing changed, so abort the loop
            break
    # Cleanup
    values.cleanup(remove=True)
    positions.cleanup(remove=True)
    return values, positions


def crust_fix(crust: ChunkGrid[np.bool8],
              outer_fill: ChunkGrid[np.bool8],
              crust_outer: ChunkGrid[np.bool8],
              crust_inner: ChunkGrid[np.bool8],
              min_distance: int = 1,
              data_pts: Optional[np.ndarray] = None,  # for plotting
              return_figs=False
              ):
    CHUNKSIZE = crust.chunk_size
    normal_kernel = make_normal_kernel()

    inv_outer_fill = ~outer_fill

    # Method cache (prevent lookup in loop)
    __grid_set_value = ChunkGrid.set_value
    __np_sum = np.sum

    figs: Dict[str, go.Figure] = dict()

    print("\tCreate Normals: ")
    with timed("\t\tTime: "):
        normal_zero = np.zeros(3, dtype=np.float32)
        normal_pos = np.array(list(crust_outer.where()))
        normal_val = np.full((len(normal_pos), 3), 0.0, dtype=np.float32)
        for n, p in enumerate(normal_pos):
            x, y, z = p
            mask: np.ndarray = outer_fill[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
            normal_val[n] = __np_sum(normal_kernel[mask], axis=0)
        normal_val = (normal_val.T / np.linalg.norm(normal_val, axis=1)).T

    print("\tGrid Normals: ")
    with timed("\t\tTime: "):
        normals: ChunkGrid[np.float32] = ChunkGrid(CHUNKSIZE, np.dtype((np.float32, (3,))), 0.0)
        normals[normal_pos] = normal_val

    print("\tRender Normal Propagation: ")
    with timed("\t\tTime: "):
        markers_outer = np.array(
            [v for p, n in normals.items(mask=crust_outer) for v in (p, p + n, (np.nan, np.nan, np.nan))],
            dtype=np.float32) + 0.5
        markers_outer_tips = np.array(
            [p + n for p, n in normals.items(mask=crust_outer)],
            dtype=np.float32) + 0.5

        ren = CloudRender()
        fig = ren.make_figure(title="Crust-Fix: Start Normal Propagation")
        fig.add_trace(ren.make_scatter(markers_outer, marker=dict(opacity=0.5, ), mode="lines", name="Start normal"))
        fig.add_trace(ren.make_scatter(markers_outer_tips, marker=dict(size=1, symbol='x'), name="Start nromal end"))
        if data_pts is not None:
            fig.add_trace(ren.make_scatter(data_pts, opacity=0.1, size=1, name='Model'))
        if return_figs:
            figs["normals"] = fig
        else:
            fig.show()

    print("\tNormal Propagation")
    with timed("\t\tTime: "):
        iterations = CHUNKSIZE
        nfield, nmask = propagate_normals(iterations, normals, crust_outer, inv_outer_fill)
        field_reset_mask = outer_fill ^ crust_outer
        nfield[field_reset_mask] = 0
        nmask[field_reset_mask] = False
        nfield.cleanup(remove=True)
        nmask.cleanup(remove=True)

    # print("\tRender Normal Field: ")
    # with timed("\t\tTime: "):
    #
    #     markers_crust = np.array(
    #         [v for p, n in nfield.items(mask=crust) for v in (p, p + n, (np.nan, np.nan, np.nan))],
    #         dtype=np.float32) + 0.5
    #     markers_outer = np.array(
    #         [v for p, n in nfield.items(mask=crust_outer) for v in (p, p + n, (np.nan, np.nan, np.nan))],
    #         dtype=np.float32) + 0.5
    #     markers_outer_tips = np.array(
    #         [p + n for p, n in nfield.items(mask=crust_outer)],
    #         dtype=np.float32) + 0.5
    #
    #     ren = CloudRender()
    #     fig = ren.make_figure(title="Crust-Fix: Normal Field")
    #     fig.add_trace(ren.make_scatter(markers_outer, marker=dict(opacity=0.5, ), mode="lines", name="Start normal"))
    #     fig.add_trace(ren.make_scatter(markers_outer_tips, marker=dict(size=1, symbol='x'), name="Start normal end"))
    #     fig.add_trace(ren.make_scatter(markers_crust, marker=dict(opacity=0.5, ), mode="lines", name="Normal field"))
    #     # fig.add_trace(VoxelRender().grid_voxel(nmask, opacity=0.1, name="Normal mask"))
    #     if data_pts is not None:
    #         fig.add_trace(ren.make_scatter(data_pts, opacity=0.1, size=1, name='Model'))
    #     fig.show()

    print("\tNormal cone: ")
    with timed("\t\tTime: "):
        medial = ChunkGrid(crust.chunk_size, np.bool8, False)
        cone_threshold: float = 0.5 * np.pi
        min_norm: float = 1e-15
        for chunk in nfield.chunks:
            padded = nfield.padding_at(chunk.index, 1, corners=True, edges=True)
            padded_mask = nmask.padding_at(chunk.index, 1, corners=True, edges=True)
            cones = normal_cone_angles(padded, padded_mask, cone_threshold, min_norm)
            medial.ensure_chunk_at_index(chunk.index).set_array(cones)

    print("\tResult: ")
    with timed("\t\tTime: "):
        # Remove artifacts where the inner and outer crusts are touching
        artifacts_fix = outer_fill.copy().pad_chunks(1)
        artifacts_fix.fill_value = False
        artifacts_fix = ~dilate(artifacts_fix, steps=max(1, min_distance) + 2) & ~outer_fill
        medial_cleaned = medial & artifacts_fix
        medial_cleaned.cleanup(remove=True)

    print("\tRender 2: ")
    with timed("\t\tTime: "):
        ren = VoxelRender()
        fig = ren.make_figure(title="Crust-Fix: Result")
        print("Ren2-medial")
        fig.add_trace(ren.grid_voxel(medial, opacity=0.3, name='Medial'))
        # fig.add_trace(ren.grid_voxel(medial_cleaned, opacity=0.05, name='Fixed'))
        print("Ren2-crust_outer")
        fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.05, name='Outer'))
        if data_pts is not None:
            print("Ren2-data_pts")
            fig.add_trace(CloudRender().make_scatter(data_pts, opacity=0.2, size=1, name='Model'))
        print("Ren2-show")
        if return_figs:
            figs["medial_axis"] = fig
        else:
            fig.show()

    print("\tRender 3: ")
    with timed("\t\tTime: "):
        ren = VoxelRender()
        fig = ren.make_figure(title="Crust-Fix: Result")
        # fig.add_trace(ren.grid_voxel(medial, opacity=0.3, name='Fixed'))
        print("Ren2-medial_cleaned")
        fig.add_trace(ren.grid_voxel(medial_cleaned, opacity=0.3, name='Medial-Cleaned'))
        print("Ren3-crust_outer")
        fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.05, name='Outer'))
        if data_pts is not None:
            print("Ren3-data_pts")
            fig.add_trace(CloudRender().make_scatter(data_pts, opacity=0.2, size=1, name='Model'))
        print("Ren3-show")
        if return_figs:
            figs["medial_axis_cleaned"] = fig
        else:
            fig.show()

    if return_figs:
        return medial_cleaned, figs
    return medial_cleaned
