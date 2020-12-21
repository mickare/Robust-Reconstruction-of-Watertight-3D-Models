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

    resolution = 64

    crust = ChunkGrid(16, dtype=bool, fill_value=False)
    scaled = (data - data_min) * resolution / data_delta_max
    assert scaled.shape[1] == 3

    crust[scaled] = True

    color = ChunkGrid(16, dtype=int, fill_value=0)
    color[crust] = 1

    extra = set(tuple(n) for i in color.chunks.keys() for f, n in color.iter_neighbors_indicies(i))
    for e in extra:
        color.ensure_chunk_at_index(e)

    components = 2

    while components > 1:
        dilated = dilate(color == 1)
        color[dilated] = 1

        colors = 2
        while True:
            start = np.array([])
            for i, c in color.chunks.items():
                if not c.all():
                    start = np.argwhere(np.logical_not(c.to_array()))[0] + c.index * color.chunk_size
                    break

            if start.size == 0:
                break

            fill_mask = flood_fill_at(start, color == 0)
            color[fill_mask] = colors
            colors += 1

        components = colors - 1
        print(components)
        ren = VoxelRender()
        fig = ren.make_figure()
        fig.add_trace(ren.grid_voxel(crust, opacity=0.5, flatshading=True, name='crust'))
        for c in range(colors):
            fig.add_trace(ren.grid_voxel(color == c, opacity=0.2, flatshading=True, name=str(c)))
        fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
        fig.show()
        color = color == 1
        color[color] = 1
