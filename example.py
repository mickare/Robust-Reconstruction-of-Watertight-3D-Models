import numpy as np

from data.chunks import ChunkGrid, Chunk
from model.model_pts import PtsModelLoader
from operators.dilate_operator import dilate
from operators.fill_operator import flood_fill_at
from render_cloud import CloudRender
from render_voxel import VoxelRender

if __name__ == '__main__':
    data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    # data = PlyModelLoader().load("models/dragon_stand/dragonStandRight.conf")
    # data = MeshModelLoader(samples=30000, noise=0.1).load("models/cat/cat_reference.obj")

    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    data_delta_max = np.max(data_max - data_min)

    resolution = 64

    grid = ChunkGrid(16, dtype=int, fill_value=0)
    scaled = (data - data_min) * resolution / data_delta_max
    assert scaled.shape[1] == 3

    grid[scaled] = 1
    grid.pad_chunks(1)

    dilated = dilate(grid == 1, steps=3)
    grid[dilated & (grid != 1)] = 2

    padded = grid == 0

    outer = next(grid.hull())
    fill_mask = flood_fill_at(outer.index * padded.chunk_size, grid == 0)
    grid[fill_mask] = 3

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(grid == 1, opacity=0.5, flatshading=True, name="Crust"))
    fig.add_trace(ren.grid_voxel(grid == 2, opacity=0.2, flatshading=True, name="Dilated"))
    fig.add_trace(ren.grid_voxel(grid == 3, opacity=0.05, flatshading=True, name="Fill"))
    # fig.add_trace(ren.grid_voxel(padded, opacity=1.0, flatshading=True, name="Padded"))
    # array, offset = (grid == 1).to_sparse()
    # fig.add_trace(ren.dense_voxel(array.todense(), offset=offset+(0,0,20), opacity=0.5, flatshading=True))
    # fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
    fig.show()
