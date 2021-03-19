"""
Steps and methods used during model reconstruction
"""
from typing import Optional, Tuple

import numba
import numpy as np
from scipy import ndimage

from reconstruction.data.chunks import ChunkGrid, Chunk
from reconstruction.filters.dilate import dilate
from reconstruction.filters.fill import flood_fill_at
from reconstruction.mathlib import Vec3i, Vec3f
from reconstruction.render.voxel_render import VoxelRender

numba.config.THREADING_LAYER = 'omp'


# =====================================================================
# Model Loading
# =====================================================================
def scale_model(model: np.ndarray, resolution=64) -> Tuple[np.ndarray, Vec3f, float]:
    assert model.ndim == 2 and model.shape[1] == 3

    model_min, model_max = np.min(model, axis=0), np.max(model, axis=0)
    model_delta_max = np.max(model_max - model_min)
    scale_factor = resolution / model_delta_max
    scaled = (model - model_min) * scale_factor

    return scaled, model_min, scale_factor


# =====================================================================
# Crust Dilation
# =====================================================================
def find_empty_point_in_chunk(mask: Chunk[np.bool8]) -> Optional[Vec3i]:
    """
    Find an empty voxel in a chunk
    :param mask: voxel chunk
    :return: global voxel position or None
    """
    if mask.is_filled():
        if mask.value:
            return mask.position_low
    else:
        pt = np.argwhere(mask.to_array())
        if len(pt) > 0:
            return pt[0] + mask.position_low
    return None


def find_empty_fill_position(mask: ChunkGrid[np.bool8]) -> Optional[Vec3i]:
    """
    Find an empty voxel in a grid
    :param mask: voxel grid
    :return: global voxel position or None
    """
    for i, c in mask.chunks.items():
        if c.any():
            return find_empty_point_in_chunk(c)
    return None


def points_on_chunk_hull(mask: ChunkGrid[np.bool8], count: int = 1) -> Optional[np.ndarray]:
    """
    Find an empty voxel in the outer chunks of a grid
    :param mask: voxel grid
    :param count: array of global voxel positions or None
    :return: global voxel positions
    """
    if not mask.chunks:
        return None

    pts_iter = (c.position_low for c in mask.iter_hull() if c.is_filled() and not c.value)
    pts = []
    for p, _ in zip(pts_iter, range(count)):
        pts.append(p)
    if pts:
        return np.asarray(pts, dtype=int)
    else:
        for c in mask.iter_hull():
            if c.any():
                p = find_empty_point_in_chunk(c)
                if p is not None:
                    return p
    return None


def fill_components(crust: ChunkGrid[np.bool8], max_components=4) -> Tuple[ChunkGrid[np.int8], int]:
    """
    Detect the components that are seperated by a crust voxel model by flood filling empty voxels.
    :param crust: the crust voxels
    :param max_components: maximum number of components
    :return: component voxels
    """
    assert not crust.fill_value
    components = crust.copy(dtype=np.int8, fill_value=np.int8(0))
    count = 1
    target_fill = points_on_chunk_hull(~crust)
    while target_fill is not None:
        count += 1
        if count > max_components:
            break
        fill_mask = flood_fill_at(target_fill, mask=components == 0)
        assert fill_mask._fill_value == (count == 2)  # Error when the outer fill does not set the _fill_value

        components[fill_mask] = count
        # Update for next iteration
        target_fill = find_empty_fill_position(components == 0)
    return components, count


def crust_dilation(crust: ChunkGrid[np.bool8], max_components=5, reverse_steps=3, max_steps=5):
    """Dilate a crust until the inner component vanishes and return result of some steps reversed"""
    assert max_steps > 0
    max_count = 0
    dilation_step = 0
    crusts_all = []
    components_all = []
    counts_all = []

    for dilation_step in range(max_steps):
        print(f"\t\tDilation-Step {dilation_step}")
        components, count = fill_components(crust, max_components=max_components)
        crusts_all.append(crust)
        components_all.append(components)
        counts_all.append(count)

        # plot_voxels(components == 0, components)
        # print(count)

        if max_count >= count and count == 2:
            break
        else:
            max_count = max(max_count, count)
            crust = dilate(crust)
            assert crust.any()

    print("\tSteps: ", dilation_step)

    # Take the crust one step before the inner component vanished.

    step = max(0, dilation_step - reverse_steps)
    crust = crusts_all[step]
    components = components_all[step]
    count_prev = counts_all[step]
    crust.cleanup(remove=True)
    components.cleanup(remove=True)

    # Cleanup components and select only the largest component
    # This will set all other components (0, 3,4,5,...) to be part of crust (1)
    cleanup_components(crust, components, count_prev)

    return crust, components, step


def cleanup_components(crust: ChunkGrid[np.bool8], components: ChunkGrid[np.int8], count: int):
    """
    Cleanup components and select only the largest component
    This will set all other components (0, 3,4,5,...) to be part of crust (1)
    """
    candidates = dict()
    candidates[0] = (components == 0).sum()
    for c in range(3, count):
        candidates[c] = (components == c).sum()

    winner_index, winner_value = max(list(candidates.items()), key=lambda e: e[1])
    for cid, cval in candidates.items():
        if cid != winner_index:
            selection = components == cid
            components[selection] = 1  # Make it a crust
            crust[selection] = True


# =====================================================================
# Distance Diffusion
# =====================================================================

def diffuse(model: ChunkGrid[bool], repeat=1):
    """
    Diffuse the voxels in model to their neighboring voxels
    :param model: the model to diffuse
    :param repeat: number of diffusion steps
    :return: diffused model
    """
    kernel = np.zeros((3, 3, 3), dtype=float)
    kernel[1] = 1
    kernel[:, 1] = 1
    kernel[:, :, 1] = 1
    kernel /= np.sum(kernel)

    result = ChunkGrid(model.chunk_size, dtype=float, fill_value=1.0)
    result[model] = 0.0
    result.pad_chunks(repeat // result.chunk_size + 1)

    for r in range(repeat):
        tmp = result.copy(empty=True)
        for chunk in result.chunks:
            padded = chunk.padding(result, 1)
            ndimage.convolve(padded, kernel, output=padded, mode='constant', cval=1.0)
            conv = padded[1:-1, 1:-1, 1:-1]
            m = model.ensure_chunk_at_index(chunk.index, insert=False)
            if m.is_filled():
                if m.value:
                    tmp.ensure_chunk_at_index(chunk.index).set_fill(0.0)
                    continue
            else:
                conv[m.to_array()] = 0.0
            tmp.ensure_chunk_at_index(chunk.index).set_array(conv)
            # Expand chunks
            for f, i in ChunkGrid.iter_neighbors_indices(chunk.index):
                tmp.ensure_chunk_at_index(i)

        result = tmp

    result.cleanup(remove=True)
    return result


# =====================================================================
# Render
# =====================================================================
def plot_voxels(grid: ChunkGrid[np.bool8], components: ChunkGrid[np.int8], title: Optional[str] = None):
    """
    Plot a voxel model
    :param grid: main voxel model
    :param components: crust components
    :param title: String title (optional)
    :return: plotly figure, to show use `<return>.show()`
    """
    ren = VoxelRender()
    fig = ren.make_figure(title=title)
    fig.add_trace(ren.grid_voxel(grid, opacity=1.0, name=f"Missing"))
    # fig.add_trace(ren.grid_voxel(components == 1, opacity=0.2, name=f"Crust"))
    fig.add_trace(ren.grid_voxel(components == 2, opacity=0.1, name=f"Hull"))
    unique = components.unique()
    for c in unique:
        if c > 2:
            fig.add_trace(ren.grid_voxel(components == c, opacity=1.0, name=f"Component {c}"))
    fig.update_layout(showlegend=True)
    return fig
