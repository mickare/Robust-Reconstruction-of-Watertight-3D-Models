from typing import Optional, Tuple, Dict, List

import numpy as np
import tqdm
from scipy import signal
from scipy.sparse import dok_matrix

from data.chunks import ChunkGrid, Chunk, ChunkFace
from filters.dilate import dilate
from filters.fill import flood_fill_at
from mathlib import Vec3i, Vec3f
from model.model_pts import FixedPtsModels
from render_cloud import CloudRender
from render_voxel import VoxelRender
from utils import timed

import maxflow

CHUNKSIZE = 8

# =====================================================================
# Model Loading
# =====================================================================
print("Loading model")


def scale_model(model: np.ndarray, resolution=64) -> Tuple[np.ndarray, Vec3f, float]:
    assert model.ndim == 2 and model.shape[1] == 3

    model_min, model_max = np.min(model, axis=0), np.max(model, axis=0)
    model_delta_max = np.max(model_max - model_min)
    scale_factor = resolution / model_delta_max
    scaled = (model - model_min) * scale_factor

    return scaled, model_min, scale_factor


resolution = 64

with timed("\tTime: "):
    data = FixedPtsModels.bunny()
    data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
    model: ChunkGrid[np.bool8] = ChunkGrid(CHUNKSIZE, dtype=np.bool8, fill_value=np.bool8(False))
    model[data_pts] = True
    model.pad_chunks(2)
    model.cleanup()

# =====================================================================
# Crust Dilation
# =====================================================================
print("Dilation")


def find_empty_point_in_chunk(chunk: Chunk[np.bool8]) -> Optional[Vec3i]:
    if chunk.is_filled():
        if chunk.value:
            return chunk.position_low
    else:
        pt = np.argwhere(chunk.to_array())
        if len(pt) > 0:
            return pt[0] + chunk.position_low
    return None


def find_empty_fill_position(mask: ChunkGrid[np.bool8]) -> Optional[Vec3i]:
    for i, c in mask.chunks.items():
        if c.any():
            return find_empty_point_in_chunk(c)
    return None


def points_on_chunk_hull(grid: ChunkGrid[np.bool8], count: int = 1) -> Optional[np.ndarray]:
    if not grid.chunks:
        return None

    pts_iter = (c.position_low for c in grid.iter_hull() if c.is_filled() and not c.value)
    pts = []
    for p, _ in zip(pts_iter, range(count)):
        pts.append(p)
    if pts:
        return np.asarray(pts, dtype=int)
    else:
        for c in grid.iter_hull():
            if c.any():
                p = find_empty_point_in_chunk(c)
                if p is not None:
                    return p
    return None


def plot_voxels(grid: ChunkGrid[np.bool8], components: ChunkGrid[int]):
    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(grid, opacity=1.0, name=f"Something"))
    fig.add_trace(ren.grid_voxel(components == 1, opacity=0.2, name=f"Crust"))
    fig.add_trace(ren.grid_voxel(components == 2, opacity=0.1, name=f"Hull 2"))
    fig.update_layout(showlegend=True)
    fig.show()


with timed("\tTime: "):
    max_count = 4
    last_count = 0
    crust: ChunkGrid[np.bool8] = model.copy()
    dilation_step = 0

    crusts_all = []
    components_all = []

    for dilation_step in range(30):
        components = crust.copy(dtype=np.int8)
        count = 1

        crusts_all.append(crust)
        components_all.append(components)

        # Find fill components
        target_fill = points_on_chunk_hull(crust)

        while target_fill is not None:
            count += 1
            if count > max_count:
                break

            fill_mask = flood_fill_at(target_fill, mask=components == 0)
            components[fill_mask] = count
            # Update for next iteration
            target_fill = find_empty_fill_position(components == 0)

        # plot_voxels(components == 0, components)

        if last_count >= count and count <= 3:
            break
        else:
            last_count = count
            crust = dilate(crust)

crust = crusts_all[max(0, len(crusts_all) - 2)]
components = components_all[max(0, len(components_all) - 2)]
crust.cleanup()
components.cleanup()

crust_outer = dilate(components == 2) & crust
crust_inner = dilate(components == 3) & crust

print("\tSteps: ", dilation_step)

# =====================================================================
# Distance Diffusion
# =====================================================================
print("Diffusion")


def diffuse(model: ChunkGrid[bool], repeat=1):
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
            conv = signal.convolve(chunk.padding(result, 1), kernel, mode='valid', method='direct')
            conv[model.ensure_chunk_at_index(chunk.index).to_array()] = 0.0
            tmp.ensure_chunk_at_index(chunk.index).set_array(conv)
        result = tmp

    result.cleanup()
    return result


with timed("\tTime: "):
    diff = diffuse(model, repeat=dilation_step)

# =====================================================================
# MinCut
# =====================================================================
print("MinCut")

# weights = ChunkGrid(diff.chunk_size, dtype=float, fill_value=0)
weights = (diff ** 4) + 1e-05
weights[~crust] = 0
weights.cleanup(remove=True)


def get_node(pos: Vec3i, face: ChunkFace) -> Tuple[Vec3i, ChunkFace]:
    """Basically forces to have only positive-direction faces"""
    if face % 2 == 0:
        return pos, face
    else:
        return tuple(np.add(pos, face.direction, dtype=int)), face.flip()


voxels = {tuple(p): w for p, w in weights.items(mask=crust)}
nodes = list(set(get_node(p, f) for p in voxels.keys() for f in ChunkFace))
nodes_index = {f: n for n, f in enumerate(nodes)}

nodes_count = len(nodes)

graph = maxflow.Graph[float](nodes_count, nodes_count)
g_nodes = graph.add_nodes(len(nodes))

for vPos, w in tqdm.tqdm(voxels.items(), total=len(voxels), desc="Linking Faces"):
    for f in ChunkFace:  # type: ChunkFace
        fNode = get_node(vPos, f)
        fIndex = nodes_index[fNode]
        for o in f.orthogonal():
            oNode = get_node(vPos, o)
            oIndex = nodes_index[oNode]
            graph.add_edge(fIndex, oIndex, w, w)

# Source
for vPos in tqdm.tqdm(list(crust_outer.where()), desc="Linking Source"):
    assert crust.get_value(vPos)
    for f in ChunkFace:  # type: ChunkFace
        fNode = get_node(tuple(vPos), f)
        fIndex = nodes_index[fNode]
        graph.add_tedge(fIndex, 1000, 0)

# Sink
for vPos in tqdm.tqdm(list(crust_inner.where()), desc="Linking Sink"):
    assert crust.get_value(vPos)
    for f in ChunkFace:  # type: ChunkFace
        fNode = get_node(tuple(vPos), f)
        fIndex = nodes_index[fNode]
        graph.add_tedge(fIndex, 0, 1000)

flow = graph.maxflow()
segments = graph.get_grid_segments(np.arange(nodes_count))

cut = ChunkGrid(CHUNKSIZE, int, 0)
cut[[p for p, f in nodes]] = segments + 1

# =====================================================================
# Render
# =====================================================================
print("Render")

with timed("\tTime: "):
    ren = VoxelRender()
    fig = ren.make_figure()
    # fig.add_trace(ren.grid_voxel(model, opacity=0.1, name='Model'))
    # fig.add_trace(ren.grid_voxel(crust, opacity=0.1, name='Crust'))
    # fig.add_trace(ren.grid_voxel(components == 1, opacity=0.1, name='Crust'))
    # fig.add_trace(ren.grid_voxel(components == 3, opacity=0.1, name='Crust'))
    # fig.add_trace(ren.grid_voxel(components == 4, opacity=0.1, name='Crust'))

    # fig.add_trace(ren.grid_voxel(diff == 0, opacity=0.1, name='Crust'))
    scat_kwargs = dict(
        marker=dict(
            size=1.0,
            colorscale=[[0.0, 'rgb(0,0,0)'], [1.0, 'rgb(255,255,255)']],
            cmin=0.0,
            cmax=1.0
        ),
        mode="markers"
    )
    fig.add_trace(CloudRender().make_value_scatter(diff, mask=crust,
                                                   name="Diffusion", **scat_kwargs))
    fig.show()

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(cut == 1, opacity=0.2, name='Cut 1'))
    fig.add_trace(ren.grid_voxel(cut == 2, opacity=0.2, name='Cut 2'))
    fig.add_trace(CloudRender().make_scatter(data_pts, name='Model'))
    fig.show()
