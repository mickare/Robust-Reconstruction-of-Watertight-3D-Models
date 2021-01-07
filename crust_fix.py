import multiprocessing
from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.chunks import ChunkGrid, ChunkFace
from filters.dilate import dilate
from render_cloud import CloudRender
from render_voxel import VoxelRender
from utils import timed


@dataclass
class NormalCorrelationWorker:
    crust: ChunkGrid[np.bool]
    kernel: np.ndarray
    steps: int

    def run(self, field: ChunkGrid[np.float]):
        crust = self.crust
        kernel = self.kernel
        for step in range(self.steps):
            field[crust] = field.correlate(kernel, fill_value=0.0)
        return field


@dataclass
class CrustFixWorker:
    merged: ChunkGrid[np.ndarray]
    neighbor_mask: np.ndarray
    angle_threshold: float = np.pi * 0.5

    def run_all(self, positions: np.ndarray):
        __np_linalg_norm = np.linalg.norm
        __np_any = np.any
        __np_arccos = np.arccos
        merged = self.merged
        threshold = self.angle_threshold
        neighbor_mask = self.neighbor_mask

        result = np.zeros(len(positions), dtype=bool)
        for n, pos in enumerate(positions):
            x, y, z = pos
            neighbors = merged[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2][neighbor_mask]
            norm = __np_linalg_norm(neighbors, axis=1)
            if __np_any(norm):
                neighbors = (neighbors[norm > 0].T / norm[norm > 0]).T
                if len(neighbors) > 0:
                    result[n] = __np_any(__np_arccos(neighbors @ neighbors.T) > threshold)
        return positions, result


def make_normal_kernel() -> np.ndarray:
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    normals = np.full((3, 3, 3), np.zeros(3), dtype=np.ndarray)
    for f1 in ChunkFace:  # type: ChunkFace
        i1 = np.add(f1.direction, (1, 1, 1))
        d1 = np.array(f1.direction, dtype=np.float)
        normals[tuple(i1)] = d1
        for f2 in f1.orthogonal():  # type: ChunkFace
            i2 = i1 + f2.direction
            d2 = d1 + f2.direction
            normals[tuple(i2)] = d2 / sqrt2
            for f3 in f2.orthogonal():
                if f1 // 2 != f3 // 2:
                    i3 = i2 + f3.direction
                    d3 = d2 + f3.direction
                    normals[tuple(i3)] = d3 / sqrt3
    return normals


def crust_fix(crust: ChunkGrid[np.bool8],
              outer_fill: ChunkGrid[np.bool8],
              crust_outer: ChunkGrid[np.bool8],
              crust_inner: Optional[ChunkGrid[np.bool8]] = None,  # for plotting
              data_pts: Optional[np.ndarray] = None  # for plotting
              ):
    CHUNKSIZE = crust.chunk_size
    normal_kernel = make_normal_kernel()

    normals_x: ChunkGrid[np.float] = ChunkGrid(CHUNKSIZE, np.float, 0.0)
    normals_y: ChunkGrid[np.float] = ChunkGrid(CHUNKSIZE, np.float, 0.0)
    normals_z: ChunkGrid[np.float] = ChunkGrid(CHUNKSIZE, np.float, 0.0)

    # Method cache (prevent lookup in loop)
    __grid_set_value = ChunkGrid.set_value
    __np_sum = np.sum

    normal_zero = np.zeros(3, dtype=np.float)
    normal_pos = np.array(list(crust_outer.where()))
    normal_val = np.full((len(normal_pos), 3), 0.0, dtype=np.float)
    for n, p in enumerate(normal_pos):
        x, y, z = p
        mask: np.ndarray = outer_fill[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        normal_val[n] = __np_sum(normal_kernel[mask], initial=normal_zero)
    normal_val = (normal_val.T / np.linalg.norm(normal_val, axis=1)).T

    normals_x[normal_pos] = normal_val.T[0]
    normals_y[normal_pos] = normal_val.T[1]
    normals_z[normal_pos] = normal_val.T[2]

    def gkern3d(size: int, sigma=1.0):
        """From: https://stackoverflow.com/questions/45723088/how-to-blur-3d-array-of-points-while-maintaining-their-original-values-python"""
        ts = np.arange(- size, size + 1, 1)
        xs, ys, zs = np.meshgrid(ts, ts, ts)
        return np.exp(-(xs ** 2 + ys ** 2 + zs ** 2) / (2 * sigma ** 2))

    def kern_simple():
        kernel = np.zeros((3, 3, 3), dtype=np.float)
        kernel[1, :, :] = 1
        kernel[:, 1, :] = 1
        kernel[:, :, 1] = 1
        kernel[:, 1, 1] = 1
        kernel[1, :, 1] = 1
        kernel[1, 1, :] = 1
        kernel[1, 1, 1] = 1
        return kernel

    # kernel = gkern3d(1)
    kernel = kern_simple()
    kernel /= np.sum(kernel)
    nfieldx = normals_x.copy()
    nfieldy = normals_y.copy()
    nfieldz = normals_z.copy()

    nfieldx.pad_chunks(1)
    nfieldy.pad_chunks(1)
    nfieldz.pad_chunks(1)

    with timed("Normal Correlation: "):
        with multiprocessing.Pool() as pool:
            worker = NormalCorrelationWorker(crust, kernel, CHUNKSIZE)
            nfieldx, nfieldy, nfieldz = pool.map(worker.run, [nfieldx, nfieldy, nfieldz])

    with timed("Normal Stacking: "):
        merged = ChunkGrid.stack([nfieldx, nfieldy, nfieldz], fill_value=0.0)

    markers_crust = np.array(
        [(p, p + n, (np.nan, np.nan, np.nan)) for p, n in merged.items(mask=crust)]
    ).reshape((-1, 3)) + 0.5
    markers_outer = np.array(
        [(p, p + n, (np.nan, np.nan, np.nan)) for p, n in merged.items(mask=crust_outer)]
    ).reshape((-1, 3)) + 0.5

    ren = CloudRender()
    fig = ren.make_figure()
    if data_pts is not None:
        fig.add_trace(ren.make_scatter(data_pts, size=1, name='Model'))
    fig.add_trace(ren.make_scatter(
        markers_crust,
        marker=dict(
            # size=3,
            opacity=0.5,
            # symbol='x'
        ),
        mode="lines",
        name="Normals"
    ))
    fig.add_trace(ren.make_scatter(
        markers_outer,
        marker=dict(
            # size=3,
            opacity=0.5,
            # symbol='x'
        ),
        mode="lines",
        name="Initial"
    ))
    fig.show()

    # Make mask of all 26 neighbors
    neighbor_mask = np.ones((3, 3, 3), dtype=bool)
    neighbor_mask[1, 1, 1] = False

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

    __np_linalg_norm = np.linalg.norm
    crust_fixed = crust_outer.copy(empty=True)
    with multiprocessing.Pool() as pool:
        worker = CrustFixWorker(merged, neighbor_mask)
        points, normals = np.array(list(merged.items(mask=crust)), dtype=np.float).transpose((1, 0, 2))
        points = points[np.linalg.norm(normals, axis=1) >= 1e-15].astype(np.int)
        # Use numpy to split the work
        splits = len(points) // min(8192, merged.chunk_size ** 3)
        if splits > 2:
            result = pool.imap_unordered(worker.run_all, np.array_split(points, splits))
        else:
            result = [worker.run_all(points)]
        for positions, result in result:
            crust_fixed[positions] = result

    # Remove artifacts where the inner and outer crusts are touching
    crust_fixed &= ~dilate(crust_outer, steps=3)

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.2, name='Outer'))
    if crust_inner is not None:
        fig.add_trace(ren.grid_voxel(crust_inner, opacity=0.2, name='Inner'))
    fig.add_trace(ren.grid_voxel(crust_fixed, opacity=1.0, name='Fixed'))
    if data_pts is not None:
        fig.add_trace(CloudRender().make_scatter(data_pts, size=1, name='Model'))
    fig.show()

    return crust_fixed
