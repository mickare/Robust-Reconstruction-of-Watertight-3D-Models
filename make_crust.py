from typing import Optional, Tuple, List

import numpy as np

from data.chunks import ChunkGrid
from mathlib import Vec3i, Vec3f
from model.model_pts import PtsModelLoader
from operators.dilate_operator import dilate
from operators.fill_operator import flood_fill_at
from render_cloud import CloudRender
from render_voxel import VoxelRender


def scale_model(model: np.ndarray, resolution=64) -> Tuple[np.ndarray, Vec3f, float]:
    assert model.ndim == 2 and model.shape[1] == 3

    model_min, model_max = np.min(model, axis=0), np.max(model, axis=0)
    model_delta_max = np.max(model_max - model_min)
    scale_factor = resolution / model_delta_max
    scaled = (model - model_min) * scale_factor

    return scaled, model_min, scale_factor


def find_empty_fill_position(mask: ChunkGrid[bool]) -> Optional[Vec3i]:
    for i, c in mask.chunks.items():
        if c.any():
            return np.argwhere(c.to_array())[0] + c.position_low
    return None


def plot(components: ChunkGrid[int], colors: int = 0,
         model: Optional[np.ndarray] = None,
         fill_points: Optional[ChunkGrid[bool]] = None):
    ren = VoxelRender()
    fig = ren.make_figure()
    if model is not None:
        fig.add_trace(CloudRender().make_scatter(model, marker=dict(size=0.45), mode="text+markers", name="Model"))
    # fig.add_trace(ren.grid_voxel(crust, opacity=0.2, name='Crust'))
    if colors > 2:
        fig.add_trace(ren.grid_voxel(components == 2, opacity=0.1, name=f"Hull 2"))
    for c in range(3, colors):
        fig.add_trace(ren.grid_voxel(components == c, opacity=1.0, name=f"Comp {c}"))

    if fill_points is not None:
        fig.add_trace(ren.grid_voxel(fill_points, opacity=1.0, color='red', name="Fill points"))
    fig.update_layout(showlegend=True)
    fig.show()


if __name__ == '__main__':
    data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    # data = PlyModelLoader().load("models/dragon_stand/dragonStandRight.conf")
    # data = MeshModelLoader(samples=30000, noise=0.1).load("models/cat/cat_reference.obj")

    verbose = 2
    CHUNKSIZE = 16
    max_steps = 20

    model, model_offset, model_scale = scale_model(data, resolution=64)

    crust = ChunkGrid(CHUNKSIZE, dtype=bool, fill_value=False)
    crust[model] = True

    # Add a chunk layer around the model (fast-flood-fill can wrap the model immediately)
    crust.pad_chunks(1)

    # A counter of the components per step
    last_crust: ChunkGrid[bool] = crust
    last_count = 0
    for step in range(0, max_steps):
        # Initialize empty component grid
        components: ChunkGrid[int] = crust.astype(int).copy()

        # Keeping track of starting points for flood fill
        fill_points = ChunkGrid(CHUNKSIZE, dtype=bool)

        # find any outer empty chunk that was padded and use it as first fill position of component 2 (= outer fill)
        fill_position: Optional[Vec3i] = next(crust.hull()).index * CHUNKSIZE

        # Mask for filling, when empty abort!
        mask_empty = components == 0

        # Color value of the filled components
        color = 2
        while fill_position is not None:

            fill_points.set_pos(fill_position, True)

            if not mask_empty.any():
                raise ValueError("WTF")
                break

            if verbose > 2:
                print(f"c:\t{color} \tpos: {fill_position},")

            # Flood fill the position with the current color
            fill_mask = flood_fill_at(fill_position, mask=mask_empty, verbose=verbose > 5)
            old = components.copy()
            components[fill_mask] = color

            # Update mask
            mask_empty = components == 0

            # Find next fill position
            fill_position = find_empty_fill_position(mask_empty)
            color += 1  # Increment color

            # if step == 0:
            #     print("fill_position is None = ", fill_position is None)

        plot(components, color, model, fill_points)

        count = color - 1

        if verbose > 1:
            print(last_count, "->", count)
        if last_count > count and count == 2:  # 2 for (Crust and Outer-fill)
            if verbose > 0:
                print("Winner winner chicken dinner!")
            break

        last_count = count
        last_crust = crust.copy()
        crust = dilate(crust)

    #
    # while comps > 1:
    #     dilated = dilate(color == 1)
    #     color[dilated] = 1
    #
    #     colors = 2
    #     while True:
    #         start = np.array([])
    #         for i, c in color.chunks.items():
    #             if not c.all():
    #                 start = np.argwhere(np.logical_not(c.to_array()))[0] + c.index * color.chunk_size
    #                 break
    #
    #         if start.size == 0:
    #             break
    #
    #         fill_mask = flood_fill_at(start, color == 0)
    #         color[fill_mask] = colors
    #         colors += 1
    #
    #     comps = colors - 1
    #     print(comps)
    #     ren = VoxelRender()
    #     fig = ren.make_figure()
    #     fig.add_trace(ren.grid_voxel(crust, opacity=0.2, flatshading=True, name='crust'))
    #     for c in range(colors):
    #         if c == 2 or c == 1:
    #             continue
    #         fig.add_trace(ren.grid_voxel(color == c, opacity=1.0, flatshading=True, name=str(c)))
    #     fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
    #     fig.show()
    #     color[color != 1] = 0
