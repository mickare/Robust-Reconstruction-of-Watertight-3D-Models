import numpy as np

from data.chunks import ChunkGrid
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

    resolution = 32

    grid = ChunkGrid(16, dtype=int, fill_value=0)
    scaled = (data - data_min) * resolution / data_delta_max
    assert scaled.shape[1] == 3

    grid[scaled] = 1

    dilated = dilate(grid == 1) & (grid != 1)
    grid[dilated] = 2

    # Add padding
    filled = set(tuple(c.index) for c in grid.chunks)
    extra = set(tuple(n) for i in grid.chunks.keys() for f, n in grid.iter_neighbors_indicies(i))
    for e in extra:
        grid.ensure_chunk_at_index(e)

    fill_mask = flood_fill_at((1,1,1), grid == 0)
    grid[fill_mask] = 3

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(grid == 1, opacity=0.5, flatshading=True))
    fig.add_trace(ren.grid_voxel(grid == 2, opacity=0.1, flatshading=True))
    fig.add_trace(ren.grid_voxel(grid == 3, opacity=0.1, flatshading=True))
    # array, offset = (grid == 1).to_sparse()
    # fig.add_trace(ren.dense_voxel(array.todense(), offset=offset+(0,0,20), opacity=0.5, flatshading=True))
    fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
    fig.show()
