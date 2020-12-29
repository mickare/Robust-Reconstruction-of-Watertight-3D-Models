from typing import Optional, Tuple, Set, Sequence

import maxflow
import numpy as np
import tqdm
from scipy import ndimage

from data.chunks import ChunkGrid, Chunk, ChunkFace
from filters.dilate import dilate
from filters.fill import flood_fill_at
from mathlib import Vec3i, Vec3f
from model.model_pts import FixedPtsModels, PtsModelLoader
from render_cloud import CloudRender
from render_voxel import VoxelRender
from utils import timed


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


def plot_voxels(grid: ChunkGrid[np.bool8], components: ChunkGrid[np.int8]):
    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(grid, opacity=1.0, name=f"Missing"))
    # fig.add_trace(ren.grid_voxel(components == 1, opacity=0.2, name=f"Crust"))
    fig.add_trace(ren.grid_voxel(components == 2, opacity=0.1, name=f"Hull"))
    fig.add_trace(ren.grid_voxel(components > 2, opacity=1.0, name=f"More"))
    fig.update_layout(showlegend=True)
    fig.show()


def crust_dilation(model: ChunkGrid[np.bool8], max_count=4):
    last_count = 0
    ok_count = False
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

        plot_voxels(components == 0, components)

        if ok_count and count == 3:
            break
        else:
            if last_count >= count:
                ok_count = True
            last_count = count
            crust = dilate(crust)

    print("\tSteps: ", dilation_step)
    # crust = crusts_all[max(0, len(crusts_all) - 2)]
    # components = components_all[max(0, len(components_all) - 2)]
    crust = crusts_all[-1]
    components = components_all[-1]
    crust.cleanup()
    components.cleanup()
    return crust, components


# =====================================================================
# Distance Diffusion
# =====================================================================


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
            for f, i in ChunkGrid.iter_neighbors_indicies(chunk.index):
                tmp.ensure_chunk_at_index(i)

        result = tmp

    result.cleanup(remove=True)
    return result


# =====================================================================
# MinCut
# =====================================================================


def mincut(diff: ChunkGrid[float], crust: ChunkGrid[bool], s=4, a=1e-20):
    # weights = ChunkGrid(diff.chunk_size, dtype=float, fill_value=0)
    weights = (diff ** s) + a
    weights[~crust] = 0
    weights.cleanup(remove=True)

    NodeIndex = Tuple[Vec3i, ChunkFace]

    def get_node(pos: Vec3i, face: ChunkFace) -> NodeIndex:
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

    visited: Set[Tuple[Tuple[Vec3i, ChunkFace], Tuple[Vec3i, ChunkFace]]] = set()
    for vPos, w in tqdm.tqdm(voxels.items(), total=len(voxels), desc="Linking Faces"):
        for f in ChunkFace:  # type: ChunkFace
            fNode = get_node(vPos, f)
            fIndex = nodes_index[fNode]
            for o in f.orthogonal():
                oNode = get_node(vPos, o)
                oIndex = nodes_index[oNode]
                lenVis = len(visited)
                visited.add((fIndex, oIndex))
                if len(visited) != lenVis:
                    graph.add_edge(fIndex, oIndex, w, w)

    # Source
    for vPos in tqdm.tqdm(list(crust_outer.where()), desc="Linking Source"):
        assert crust.get_value(vPos)
        for f in ChunkFace:  # type: ChunkFace
            fNode = get_node(tuple(vPos), f)
            fIndex = nodes_index[fNode]
            graph.add_tedge(fIndex, 10000, 0)

    # Sink
    for vPos in tqdm.tqdm(list(crust_inner.where()), desc="Linking Sink"):
        assert crust.get_value(vPos)
        for f in ChunkFace:  # type: ChunkFace
            fNode = get_node(tuple(vPos), f)
            fIndex = nodes_index[fNode]
            graph.add_tedge(fIndex, 0, 10000)

    flow = graph.maxflow()
    segments = graph.get_grid_segments(np.arange(nodes_count))

    def to_voxel(nodeIndex: NodeIndex) -> Sequence[Vec3i]:
        pos, face = nodeIndex
        return [
            pos,
            np.asarray(face.direction) * (face.flip() % 2) + pos
        ]

    segment0 = ChunkGrid(crust.chunk_size, bool, False)
    segment0[[p for node, s in zip(nodes, segments) if s == False
              for p in to_voxel(node)]] = True
    segment1 = ChunkGrid(crust.chunk_size, bool, False)
    segment1[[p for node, s in zip(nodes, segments) if s == True
              for p in to_voxel(node)]] = True

    return segment0, segment1


# =====================================================================
# Render
# =====================================================================

CHUNKSIZE = 8
resolution = 32

print("Loading model")
with timed("\tTime: "):
    # data = FixedPtsModels.bunny()
    data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
    model: ChunkGrid[np.bool8] = ChunkGrid(CHUNKSIZE, dtype=np.bool8, fill_value=np.bool8(False))
    model[data_pts] = True
    model.pad_chunks(2)
    model.cleanup()

initial_crust = model

for resolution_step in range(0, 2):
    print(f"RESOLUTION STEP: {resolution_step}")
    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(initial_crust, opacity=0.1, name='Initial'))
    fig.add_trace(CloudRender().make_scatter(data_pts, name='Model'))
    fig.show()

    print("Dilation")
    with timed("\tTime: "):
        crust, components = crust_dilation(initial_crust)
        crust_outer = dilate(components == 2) & crust
        crust_inner = dilate((components != 1) & (components != 2)) & crust

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.2, name='Outer'))
    fig.add_trace(ren.grid_voxel(crust_inner, opacity=0.2, name='Inner'))
    fig.add_trace(CloudRender().make_scatter(data_pts, name='Model'))
    fig.show()

    print("Diffusion")
    with timed("\tTime: "):
        diff = diffuse(model, repeat=3)

    print("MinCut")
    with timed("\tTime: "):
        segment0, segment1 = mincut(diff, crust)
        thincrust = segment0 & segment1

    print("Render")
    with timed("\tTime: "):
        # ren = VoxelRender()
        # fig = ren.make_figure()
        # fig.add_trace(CloudRender().make_value_scatter(
        #     diff, mask=crust, name="Diffusion",
        #     marker=dict(
        #         size=1.0,
        #         colorscale=[[0.0, 'rgb(0,0,0)'], [1.0, 'rgb(255,255,255)']],
        #         cmin=0.0,
        #         cmax=1.0
        #     ),
        #     mode="markers"
        # ))
        # fig.show()

        ren = VoxelRender()
        fig = ren.make_figure()
        fig.add_trace(ren.grid_voxel(segment0, opacity=0.1, name='Segment 0'))
        fig.add_trace(ren.grid_voxel(segment1, opacity=0.1, name='Segment 1'))
        fig.add_trace(ren.grid_voxel(thincrust, opacity=1.0, name='Join'))
        fig.add_trace(CloudRender().make_scatter(data_pts, name='Model'))
        fig.show()

    print("Volumetric refinment")
    with timed("\tTime: "):
        initial_crust = thincrust.split(2)
        initial_crust.pad_chunks(1)
        initial_crust = dilate(initial_crust)

        resolution *= 2
        data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
        reinserted = ChunkGrid(initial_crust.chunk_size, np.bool8, fill_value=False)
        reinserted[data_pts] = True
        reinserted.pad_chunks(1)
        initial_crust |= dilate(reinserted, steps=3)
