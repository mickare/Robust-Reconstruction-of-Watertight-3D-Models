import numpy as np
from scipy.ndimage import binary_dilation

from data.chunks import ChunkGrid
from mathlib import Vec3i


def flood_fill(grid: ChunkGrid):
    for ind in grid.chunks:
        chunk = grid.get_chunk_by_index(ind)
        chunk.color = 0
    color = 1
    while flood_fill_recursive(grid, color):
        print(color)
        color += 1


def find_empty_chunk(grid: ChunkGrid[int], value: int):
    for chunk in grid.chunks:
        if chunk.empty():
            return chunk


def flood_fill_recursive(grid: ChunkGrid, color: int):
    # Find start chunk
    current_chunk = None
    for chunk in grid.chunks:
        if chunk.empty():
            current_chunk = chunk
        if chunk.type == ChunkType.ARRAY:
            if not chunk.value and not chunk.color:
                current_chunk = chunk
        else:
            c_color = grid.get_chunk_by_index(chunk).color_voxels()
            print(len(np.argwhere(np.logical_not(crust))))
            print(len(np.argwhere(np.logical_not(c_color))))
            mask = np.argwhere(np.logical_and(np.logical_not(c_color), np.logical_not(crust)))
            print(len(mask))
            if len(mask) > 0:
                grid.get_chunk_by_index(chunk).color = c_color
                c_color[mask[0]] = color
                current_chunk = chunk
        if current_chunk:
            break

    if not current_chunk:
        return False
    chunk_queue = []
    indices = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    while True:
        new_fill = flood_fill_chunk(grid, current_chunk, color)
        if new_fill:
            for i in indices:
                neighbor = np.asarray(current_chunk) + i
                if not all(0 <= j < grid.resolution for j in neighbor):
                    continue
                if not grid.get_chunk_by_index(neighbor):
                    grid.create_if_absent(g.index_to_pos(neighbor))
                if isinstance(grid.get_chunk_by_index(neighbor).crust, np.ndarray):
                    flood_fill_border(grid, current_chunk, neighbor, color)
                chunk_queue.append(neighbor)
        if chunk_queue:
            current_chunk = chunk_queue.pop()
        else:
            break
    return True


def flood_fill_chunk(grid: ChunkGrid, chunk_index: Vec3i, color: int):
    chunk = grid.get_chunk_by_index(chunk_index)
    if not isinstance(chunk.crust, np.ndarray) or not chunk.crust.any():
        if not chunk.color:
            chunk.color = color
            return True
        else:
            return False
    else:
        chunk_color = chunk.color_voxels()
        mask = np.logical_and(np.logical_not(chunk_color), np.logical_not(chunk.crust))
        filling = np.array([[[True if x == color else False for x in y] for y in z] for z in chunk_color])
        filling = binary_dilation(filling, iterations=-1, mask=mask)
        filling = np.where(filling, np.full(shape=chunk_color.shape, fill_value=color), chunk_color)
        if np.array_equal(filling, chunk_color):
            return False
        chunk.color = filling
        return True


def flood_fill_border(grid: ChunkGrid, source_chunk_index: Vec3i, target_chunk_index: Vec3i, color: int):
    def copy_border(target_index: int):
        def get_index(i, j):
            a = [i, j]
            a.insert(axis, target_index)
            return tuple(a)

        c = grid.get_chunk_by_index(target_chunk_index)
        if not c:
            c = g.create_if_absent(g.index_to_pos(target_chunk_index))
        c_color = c.color_voxels()
        for i, row in enumerate(segment):
            for j, e in enumerate(row):
                if e == color:
                    index = get_index(i, j)
                    if not c_color[index]:
                        c_color[index] = color
        c.color = c_color

    source_chunk = grid.get_chunk_by_index(source_chunk_index)
    segment = None if isinstance(source_chunk.color, np.ndarray) else np.full(ChunkShape[1:], dtype=int,
                                                                              fill_value=color)

    axis = -1
    for i in range(3):
        if source_chunk_index[i] != target_chunk_index[i]:
            axis = i
            break
    if axis == -1:
        raise RuntimeError("Indices are equal.")

    if source_chunk_index[axis] < target_chunk_index[axis]:
        segment = segment if isinstance(segment, np.ndarray) else source_chunk.color.take(0, axis)
        copy_border(-1)
    elif source_chunk_index[axis] > target_chunk_index[axis]:
        segment = segment if isinstance(segment, np.ndarray) else source_chunk.color.take(-1, axis)
        copy_border(0)
    else:
        raise RuntimeError("Invalid index.")


if __name__ == '__main__':
    from model.model_pts import PtsModelLoader
    from render_cloud import CloudRender
    from render_voxel import VoxelRender

    # data = MeshModelLoader(samples=30000, noise=0.1).load("models/cat/cat_reference.obj")
    data = PtsModelLoader().load("models/bunny/bunnyData.pts")

    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    data_delta_max = np.max(data_max - data_min)

    resolution = 32

    scaled = (data - data_min) * resolution / data_delta_max
    assert scaled.shape[1] == 3

    grid: ChunkGrid[int] = ChunkGrid(8, empty_value=0)
    for p in scaled:
        pos = np.array(p, dtype=int)
        c = grid.ensure_chunk_at_pos(pos)
        c.set_pos(pos, 1)

    # Dilation
    dilate(grid)

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.make_mesh(grid, opacity=0.4, flatshading=True, voxel_kwargs=dict(value=1)))
    # fig.add_trace(ren.make_mesh(grid, opacity=0.4, flatshading=True, voxel_kwargs=dict(value=2)))
    fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
    fig.show()

    # Flood filling
    flood_fill(g)
    pts = g.color_to_points() + 0.5
    args = [scaled]
    args.extend([i for i in pts])
    args = (i for i in args)
    fig = CloudRender().plot(*args, size=1)
    fig.show()
