from typing import Optional, Tuple, Dict

import numba
import numpy as np

from data.chunks import ChunkGrid, Chunk, ChunkIndex
from data.faces import ChunkFace
from filters.dilate import dilate
from render_cloud import CloudRender
from render_voxel import VoxelRender
from utils import timed


# @dataclass
# class CrustFixWorker:
#     merged: ChunkGrid[np.ndarray]
#     neighbor_mask: np.ndarray
#     angle_threshold: float = np.pi * 0.5
#
#     def run_all(self, positions: np.ndarray):
#         __np_linalg_norm = np.linalg.norm
#         __np_any = np.any
#         __np_arccos = np.arccos
#         merged = self.merged
#         threshold = self.angle_threshold
#         neighbor_mask = self.neighbor_mask
#
#         # Calculate the normal cone
#         result = np.zeros(len(positions), dtype=bool)
#         for n, pos in enumerate(positions):
#             x, y, z = pos
#             neighbors = merged[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2][neighbor_mask]
#             norm = __np_linalg_norm(neighbors, axis=1)
#             cond = norm > 0
#             if __np_any(cond):
#                 normalized = (neighbors[cond].T / norm[cond]).T
#                 result[n] = __np_any(__np_arccos(normalized @ normalized.T) >= threshold)
#         return positions, result


@numba.njit(parallel=True, fastmath=True)
def normal_cone_angles(normals: np.ndarray, threshold=0.5 * np.pi, min_norm: float = 1e-15):
    assert normals.ndim == 4
    assert normals.shape[0] == normals.shape[1] == normals.shape[2]
    assert normals.shape[3] == 3
    size = normals.shape[0]
    result = np.zeros((size - 2, size - 2, size - 2), dtype=np.bool8)
    for i in numba.pndindex((size - 2, size - 2, size - 2)):
        # Collect normals
        current = np.zeros((26, 3), dtype=np.float32)
        norm = np.zeros(26, dtype=np.float32)
        ci: numba.uint32 = 0
        for o in np.ndindex((3, 3, 3)):
            if o != (1, 1, 1):
                x, y, z = i[0] + o[0], i[1] + o[1], i[2] + o[2]
                current[ci, 0] = normals[x, y, z, 0]
                current[ci, 1] = normals[x, y, z, 1]
                current[ci, 2] = normals[x, y, z, 2]
                norm[ci] = np.linalg.norm(current[ci])
                ci += 1
        assert ci == 26

        # Check Threshold
        cond = norm > min_norm
        if np.any(cond):
            normalized = (current[cond].T / norm[cond]).T
            result[i[0], i[1], i[2]] = np.any(np.arccos(normalized @ normalized.T) >= threshold)
        # else:
        #     # Some normals are very small, which should mean they are close to the medial axis.
        #     result[i[0], i[1], i[2]] = True
    return result


def make_normal_kernel() -> np.ndarray:
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    normals = np.full((3, 3, 3), np.zeros(3), dtype=np.ndarray)
    for f1 in ChunkFace:  # type: ChunkFace
        i1 = np.add(f1.direction(), (1, 1, 1))
        d1 = np.array(f1.direction(), dtype=np.float)
        normals[tuple(i1)] = d1
        for f2 in f1.orthogonal():  # type: ChunkFace
            i2 = i1 + f2.direction()
            d2 = d1 + f2.direction()
            normals[tuple(i2)] = d2 / sqrt2
            for f3 in f2.orthogonal():
                if f1 // 2 != f3 // 2:
                    i3 = i2 + f3.direction()
                    d3 = d2 + f3.direction()
                    normals[tuple(i3)] = d3 / sqrt3
    return normals


@numba.njit(parallel=True)
def set_array_1d(arr: np.ndarray, pos: np.ndarray, values: np.ndarray):
    for i in numba.prange(len(pos)):
        arr[pos[i][0], pos[i][1], pos[i][2]] = values[i]


@numba.stencil(neighborhood=((-1, 1), (-1, 1), (-1, 1)))
def kernel3(a):
    return numba.float32(0.14285714285714285 * (a[0, 0, 0]
                                                + a[-1, 0, 0] + a[0, -1, 0] + a[0, 0, -1]
                                                + a[1, 0, 0] + a[0, 1, 0] + a[0, 0, 1]))


@numba.njit()
def correlate_iter(image: np.ndarray, positions: np.ndarray, values: np.ndarray, iterations: int) -> np.ndarray:
    assert len(positions) == len(values)
    shape = image.shape
    src = image
    set_array_1d(src, positions, values)

    dst = image.copy()
    for i in range(iterations):
        # _correlate(src, dst, kernel)
        kernel3(src, out=dst)
        # Swap
        src_old = src
        src = dst
        dst = src_old
        # Reset
        set_array_1d(src, positions, values)
    return src


class CorrelationTask:
    fixed_data: ChunkGrid[np.float32]
    fixed_mask: ChunkGrid[np.bool8]

    def __init__(self, fixed_data: ChunkGrid[np.float32], fixed_mask: ChunkGrid[np.bool8]):
        self.fixed_data = fixed_data.astype(np.float32)
        self.fixed_mask = fixed_mask.astype(np.bool8)

    def run(self, iterations: int) -> ChunkGrid[np.float32]:
        assert iterations >= 0
        size = self.fixed_data.chunk_size
        block_shape = (3, 3, 3)

        # Cache for initial block values that are fixed
        fixed_cache: Dict[ChunkIndex, Tuple[np.ndarray, np.ndarray]] = dict()

        def get_fixed_values(index: ChunkIndex):
            """Get the position and value array for a chunk index. Uses a cache"""
            key = tuple(index)
            entry = fixed_cache.get(key, None)
            if entry is None:
                data = Chunk.block_to_array(self.fixed_data.get_block_at(key, block_shape, ensure=True, insert=False))
                mask = Chunk.block_to_array(self.fixed_mask.get_block_at(key, block_shape, ensure=True, insert=False))
                pos = np.argwhere(mask)
                values = data[tuple(pos.T)]
                entry = (pos, values)
                fixed_cache[key] = entry
            return entry

        result = self.fixed_data.copy()
        while iterations > 0:
            step_iterations = min(iterations, self.fixed_data.chunk_size)
            iterations -= step_iterations
            if step_iterations <= 0:
                break

            result.pad_chunks(1)
            tmp = result.copy(empty=True)
            for index in result.chunks.keys():
                positions, values = get_fixed_values(index)
                image = Chunk.block_to_array(result.get_block_at(index, block_shape))

                arr = correlate_iter(image, positions, values, step_iterations)
                tmp.ensure_chunk_at_index(index).set_array(arr[size:2 * size, size:2 * size, size:2 * size])
                tmp.set_block_at(index, arr, replace=False)
            result = tmp
            # Cleanup
            result.cleanup(remove=True)
            # for index in list(result.chunks.keys()):
            #     c = self.chunks_mask.ensure_chunk_at_index(index, insert=False)
            #     if not c.any_fast():
            #         del result.chunks[index]
        return result


def crust_fix(crust: ChunkGrid[np.bool8],
              outer_fill: ChunkGrid[np.bool8],
              crust_outer: ChunkGrid[np.bool8],
              crust_inner: ChunkGrid[np.bool8],
              min_distance: int = 1,
              data_pts: Optional[np.ndarray] = None  # for plotting
              ):
    CHUNKSIZE = crust.chunk_size
    normal_kernel = make_normal_kernel()

    inv_outer_fill = ~outer_fill

    # Method cache (prevent lookup in loop)
    __grid_set_value = ChunkGrid.set_value
    __np_sum = np.sum

    print("\tCreate Normals: ")
    with timed("\t\tTime: "):
        normal_zero = np.zeros(3, dtype=np.float32)
        normal_pos = np.array(list(crust_outer.where()))
        normal_val = np.full((len(normal_pos), 3), 0.0, dtype=np.float32)
        for n, p in enumerate(normal_pos):
            x, y, z = p
            mask: np.ndarray = outer_fill[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
            normal_val[n] = __np_sum(normal_kernel[mask], initial=normal_zero)
        normal_val = (normal_val.T / np.linalg.norm(normal_val, axis=1)).T

    print("\tGrid Normals: ")
    with timed("\t\tTime: "):
        normals_x: ChunkGrid[np.float32] = ChunkGrid(CHUNKSIZE, np.float32, 0.0)
        normals_y: ChunkGrid[np.float32] = ChunkGrid(CHUNKSIZE, np.float32, 0.0)
        normals_z: ChunkGrid[np.float32] = ChunkGrid(CHUNKSIZE, np.float32, 0.0)

        normals_x[normal_pos] = normal_val.T[0]
        normals_y[normal_pos] = normal_val.T[1]
        normals_z[normal_pos] = normal_val.T[2]

    print("\tNormal Correlation")
    with timed("\t\tTime: "):
        iterations = CHUNKSIZE * 3
        with timed("\t\tX: "):
            nfieldx = CorrelationTask(normals_x, crust_outer).run(iterations)
        with timed("\t\tY: "):
            nfieldy = CorrelationTask(normals_y, crust_outer).run(iterations)
        with timed("\t\tZ: "):
            nfieldz = CorrelationTask(normals_z, crust_outer).run(iterations)

    field_reset_mask = outer_fill ^ crust_outer
    nfieldx[field_reset_mask] = 0
    nfieldy[field_reset_mask] = 0
    nfieldz[field_reset_mask] = 0
    nfieldx.cleanup(remove=True)
    nfieldy.cleanup(remove=True)
    nfieldz.cleanup(remove=True)

    print("mask_x-len:", len(nfieldx.chunks))
    print("mask_y-len:", len(nfieldy.chunks))
    print("mask_z-len:", len(nfieldz.chunks))

    print("\tNormal Stacking: ")
    with timed("\t\tTime: "):
        merged = ChunkGrid.stack([nfieldx, nfieldy, nfieldz], fill_value=0.0)
        print("merged-len:", len(merged.chunks))

    print("\tRender 1: ")
    with timed("\t\tTime: "):
        # markers_crust = np.array(
        #     [(p, p + n, (np.nan, np.nan, np.nan)) for p, n in merged.items(mask=crust)]
        # ).reshape((-1, 3)) + 0.5
        # markers_outer = np.array(
        #     [(p, p + n, (np.nan, np.nan, np.nan)) for p, n in merged.items(mask=crust_outer)]
        # ).reshape((-1, 3)) + 0.5

        markers_crust = np.array(
            [v for p, n in merged.items(mask=crust) for v in (p, p + n, (np.nan, np.nan, np.nan))],
            dtype=np.float32
        ) + 0.5
        markers_outer = np.array(
            [v for p, n in merged.items(mask=crust_outer) for v in (p, p + n, (np.nan, np.nan, np.nan))],
            dtype=np.float32
        ) + 0.5
        markers_outer_tips = np.array([p + n for p, n in merged.items(mask=crust_outer)], dtype=np.float32) + 0.5

        ren = CloudRender()
        fig = ren.make_figure(title="Crust-Fix: Normal field")
        if data_pts is not None:
            fig.add_trace(ren.make_scatter(data_pts, opacity=0.1, size=1, name='Model'))
        fig.add_trace(ren.make_scatter(
            markers_crust,
            marker=dict(
                opacity=0.5,
            ),
            mode="lines",
            name="Normals"
        ))
        fig.add_trace(ren.make_scatter(
            markers_outer,
            marker=dict(
                opacity=0.5,
            ),
            mode="lines",
            name="Initial"
        ))
        fig.add_trace(ren.make_scatter(
            markers_outer_tips,
            marker=dict(
                size=1,
                symbol='x'
            ),
            name="Initial end"
        ))
        fig.show()

    print("\tNormal cone: ")
    with timed("\t\tTime: "):
        # Make mask of all 26 neighbors
        # neighbor_mask = np.ones((3, 3, 3), dtype=bool)
        # neighbor_mask[1, 1, 1] = False

        # crust_fixed = crust_inner.copy(empty=True)
        # for p, n in merged.items(mask=crust):
        #     if __np_linalg_norm(n) < 1e-15:  # Skip small
        #         continue
        #     x, y, z = p
        #     neighbors = merged[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2][neighbor_mask]
        #     norm = __np_linalg_norm(neighbors, axis=1)
        #     neighbors = (neighbors[norm > 0].T / norm[norm > 0]).T
        #     found = np.argwhere(np.arccos(neighbors @ neighbors.T) > np.pi / 2)
        #     if len(found) > 0:
        #         crust_fixed.set_value(p, True)

        # __np_linalg_norm = np.linalg.norm

        crust_fixed = crust_outer.copy(empty=True)
        for chunk in merged.chunks:
            if not inv_outer_fill.ensure_chunk_at_index(chunk.index, insert=False).any():
                continue
            padded = chunk.padding(merged, 1, corners=True)
            cones = normal_cone_angles(padded, threshold=0.5 * np.pi)
            crust_fixed.ensure_chunk_at_index(chunk.index).set_array(cones)

    print("\tResult: ")
    with timed("\t\tTime: "):
        # Remove artifacts where the inner and outer crusts are touching
        mask_fix = outer_fill.copy().pad_chunks(1)
        mask_fix.fill_value = False
        mask_fix = ~dilate(mask_fix, steps=max(1, min_distance - 1)) & ~outer_fill
        crust_fixed &= mask_fix
        crust_fixed.cleanup(remove=True)

    print("\tRender 2: ")
    with timed("\t\tTime: "):
        ren = VoxelRender()
        fig = ren.make_figure(title="Crust-Fix: Result")
        fig.add_trace(ren.grid_voxel(crust_fixed, opacity=1.0, name='Fixed'))
        fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.2, name='Outer'))
        # if crust_inner is not None:
        #     fig.add_trace(ren.grid_voxel(crust_inner, opacity=0.2, name='Inner'))
        # if data_pts is not None:
        #     fig.add_trace(CloudRender().make_scatter(data_pts, size=1, name='Model'))
        fig.show()

    return crust_fixed
