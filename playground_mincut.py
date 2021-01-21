from typing import Optional, Tuple, Sequence

import numba
import numpy as np
import pytorch3d.structures
import torch
from scipy import ndimage

import mesh_extraction
from crust_fix2 import crust_fix
from data.chunks import ChunkGrid, Chunk
from filters.dilate import dilate
from filters.fill import flood_fill_at
from mathlib import Vec3i, Vec3f
from mincut import MinCut
from model.bunny import FixedBunny
from model.dragon import Dragon
from render.cloud_render import CloudRender
from render.voxel_render import VoxelRender
from utils import timed

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
    if mask.is_filled():
        if mask.value:
            return mask.position_low
    else:
        pt = np.argwhere(mask.to_array())
        if len(pt) > 0:
            return pt[0] + mask.position_low
    return None


def find_empty_fill_position(mask: ChunkGrid[np.bool8]) -> Optional[Vec3i]:
    for i, c in mask.chunks.items():
        if c.any():
            return find_empty_point_in_chunk(c)
    return None


def points_on_chunk_hull(mask: ChunkGrid[np.bool8], count: int = 1) -> Optional[np.ndarray]:
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


def plot_voxels(grid: ChunkGrid[np.bool8], components: ChunkGrid[np.int8], title: Optional[str] = None):
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
    fig.show()


def fill_components(crust: ChunkGrid[np.bool8], max_components=4) -> Tuple[ChunkGrid[np.int8], int]:
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
    # Cleanup components and select only the largest component
    # This will set all other components (0, 3,4,5,...) to be part of crust (1)
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


#
# def mincut(diff: ChunkGrid[float], crust: ChunkGrid[bool], crust_outer: ChunkGrid[bool],
#            crust_inner: ChunkGrid[bool], s=4, a=1e-20):
#     # weights = ChunkGrid(diff.chunk_size, dtype=float, fill_value=0)
#     weights = (diff ** s) + a
#     weights[~crust] = -1
#     weights.cleanup(remove=True)
#
#     NodeIndex = Tuple[Vec3i, ChunkFace]
#     CF = ChunkFace
#
#     def get_node(pos: Vec3i, face: ChunkFace) -> NodeIndex:
#         """Basically forces to have only positive-direction faces"""
#         if face % 2 == 0:
#             return pos, face
#         else:
#             return tuple(np.add(pos, face.direction(), dtype=int)), face.flip()
#
#     voxels = {tuple(p): w for p, w in weights.items(mask=crust) if w >= 0}
#     nodes = list(set(get_node(p, f) for p in voxels.keys() for f in ChunkFace))
#     nodes_index = {f: n for n, f in enumerate(nodes)}
#
#     nodes_count = len(nodes)
#
#     graph = maxflow.Graph[float](nodes_count, nodes_count)
#     g_nodes = graph.add_nodes(len(nodes))
#
#     # visited: Set[Tuple[Tuple[Vec3i, CF], Tuple[Vec3i, CF]]] = set()
#     for vPos, w in tqdm.tqdm(voxels.items(), total=len(voxels), desc="Linking Faces"):
#         iN = nodes_index[get_node(vPos, CF.NORTH)]
#         iS = nodes_index[get_node(vPos, CF.SOUTH)]
#         iT = nodes_index[get_node(vPos, CF.TOP)]
#         iB = nodes_index[get_node(vPos, CF.BOTTOM)]
#         iE = nodes_index[get_node(vPos, CF.EAST)]
#         iW = nodes_index[get_node(vPos, CF.WEST)]
#         for f, o in [
#             (iN, iE), (iN, iW), (iN, iT), (iN, iB),
#             (iS, iE), (iS, iW), (iS, iT), (iS, iB),
#             (iT, iE), (iT, iW), (iB, iE), (iB, iW)
#         ]:  # type: ChunkFace
#             graph.add_edge(f, o, w, w)
#
#     # Source
#     for vPos in tqdm.tqdm(list(crust_outer.where()), desc="Linking Source"):
#         for f in ChunkFace:  # type: ChunkFace
#             fNode = get_node(tuple(vPos), f)
#             fIndex = nodes_index.get(fNode, None)
#             if fIndex is not None:
#                 graph.add_tedge(fIndex, 10000, 0)
#
#     # Sink
#     for vPos in tqdm.tqdm(list(crust_inner.where()), desc="Linking Sink"):
#         for f in ChunkFace:  # type: ChunkFace
#             fNode = get_node(tuple(vPos), f)
#             fIndex = nodes_index.get(fNode, None)
#             if fIndex is not None:
#                 graph.add_tedge(fIndex, 0, 10000)
#
#     flow = graph.maxflow()
#     segments = graph.get_grid_segments(np.arange(nodes_count))
#
#     segment0 = ChunkGrid(crust.chunk_size, bool, False)
#     segment0[[p for node, s in zip(nodes, segments) if s == False
#               for p in to_voxel(node)]] = True
#     segment1 = ChunkGrid(crust.chunk_size, bool, False)
#     segment1[[p for node, s in zip(nodes, segments) if s == True
#               for p in to_voxel(node)]] = True
#
#     return segment0, segment1, segments, voxels, nodes_index


# =====================================================================
# Render
# =====================================================================
if __name__ == '__main__':
    CHUNKSIZE = 16
    resolution = 64

    print("Loading model")
    with timed("\tTime: "):
        data = FixedBunny.bunny()
        dilations_max = 5
        dilations_reverse = 1

        # data = Dragon().load()
        # dilations_max = 20
        # dilations_reverse = 3

        data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
        model: ChunkGrid[np.bool8] = ChunkGrid(CHUNKSIZE, dtype=np.bool8, fill_value=np.bool8(False))
        model[data_pts] = True
        model.pad_chunks(2)
        model.cleanup()

    plot_model: Optional[np.ndarray] = data_pts[::5]

    crust: ChunkGrid[np.bool8] = model.copy()
    crust.cleanup(remove=True)

    # ren = VoxelRender()
    # fig = ren.make_figure()
    # fig.add_trace(ren.grid_voxel(initial_crust, opacity=0.1, name='Initial'))
    # fig.add_trace(CloudRender().make_scatter(data_pts, size=1, name='Model'))
    # fig.show()

    print("Dilation")
    with timed("\tTime: "):
        crust, components, dilation_step = crust_dilation(crust, max_steps=dilations_max,
                                                          reverse_steps=dilations_reverse)
        # assert components._fill_value == 2

        plot_voxels(components == 0, components, title=f"Initial Dilation")
        crust_dilate = dilate(crust)
        outer_fill = components == 2
        crust_outer = outer_fill & crust_dilate
        crust_inner = (components == 3) & crust_dilate

        assert crust_dilate._fill_value == False
        assert outer_fill._fill_value == True
        assert crust_outer._fill_value == False
        assert crust_inner._fill_value == False

    """
           Approximate Voxel near Medial Axis, by propagating a Normal field inwards.
           Then for each voxel compute a normal cone and mark the voxel as inner component when the cone angle is greater than 90°.
           """
    print("Crust-Fix")
    # with timed("\tTime: "):
    #     crust_inner |= crust_fix(
    #         crust, outer_fill, crust_outer, crust_inner,
    #         min_distance=dilation_step,
    #         data_pts=plot_model
    #     )
    #     # crust_inner[model] = False  # Remove model voxels if they have been added by the crust fix

    """
    Increase resolution and make the crust_fixmesh approximation finer
    """
    for resolution_step in range(0, 5):
        print(f"RESOLUTION STEP: {resolution_step}")

        print("Render Crust")
        with timed("\tTime: "):
            ren = VoxelRender()
            fig = ren.make_figure(title=f"Step-{resolution_step}: Crust")
            fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.1, name='Outer'))
            fig.add_trace(ren.grid_voxel(crust_inner, opacity=1.0, name='Inner'))
            if plot_model is not None:
                fig.add_trace(CloudRender().make_scatter(plot_model, size=0.7, name='Model'))
            fig.show()

        print("Diffusion")
        with timed("\tTime: "):
            diff = diffuse(model, repeat=3)

        print("Plot-Diffusion")
        with timed("\tTime: "):
            ren = CloudRender()
            fig = ren.make_figure()

            # Cut in half
            diff_mask = (diff != 1.0) & crust
            half = (np.max(data_pts, axis=0) + np.min(data_pts, axis=0)).astype(int) // 2
            half_chunk = half // diff_mask.chunk_size
            half_chunk_split = half[2] % diff_mask.chunk_size
            for index in list(diff_mask.chunks.keys()):
                if index[2] > half_chunk[2]:
                    del diff_mask.chunks[index]
                elif index[2] == half_chunk[2]:
                    ch = diff_mask.chunks.get(index)
                    arr = ch.to_array()
                    arr[:, :, half_chunk_split:] = False
                    ch.set_array(arr)

            items = list(diff.items(mask=diff_mask))
            items.sort(key=lambda e: e[0][2] * 1024 + e[0][1] + e[0][0])
            points, values = zip(*items)  # type: Sequence[Vec3i], Sequence
            pts = np.array(points, dtype=np.float32) + 0.5

            fig.add_trace(ren.make_scatter(
                pts,
                name="Diffusion",
                marker=dict(
                    size=2.0,
                    opacity=0.7,
                    colorscale='Viridis',
                    color=np.array(values)
                ),
                mode="markers",
            ))
            fig.show()

        print("MinCut")
        with timed("\tTime: "):
            mincut = MinCut(diff, crust, crust_outer, crust_inner)
            segment0, segment1 = mincut.grid_segments()
            thincrust = segment0 & segment1

        print("Render")
        with timed("\tTime: "):
            ren = VoxelRender()
            fig = ren.make_figure(title=f"Step-{resolution_step}: Segments")
            fig.add_trace(ren.grid_voxel(segment0, opacity=0.1, name='Segment 0'))
            fig.add_trace(ren.grid_voxel(segment1, opacity=0.1, name='Segment 1'))
            fig.add_trace(ren.grid_voxel(thincrust, opacity=1.0, name='Join'))
            if plot_model is not None:
                fig.add_trace(CloudRender().make_scatter(plot_model, size=1, name='Model'))
            fig.show()

        print("Volumetric refinement")
        with timed("\tTime: "):
            # Rebuild model
            resolution *= 2
            data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
            model = ChunkGrid(CHUNKSIZE, np.bool8, fill_value=np.bool8(False))
            model[data_pts] = np.bool8(True)

            plot_model: Optional[np.ndarray] = data_pts[::5]

            # Build new crust
            crust = dilate(dilate(thincrust.split(2), steps=1) | dilate(model, steps=3))

            components, count = fill_components(crust, max_components=5)
            cleanup_components(crust, components, count)

            outer_fill = (components == 2)
            outer_fill.cleanup(remove=True)

            crust_dilate = dilate(crust)
            crust_outer = outer_fill & crust_dilate
            crust_inner = (components == 3) & crust_dilate

            dilation_step = 2

            # Validate data
            assert crust._fill_value == False
            assert outer_fill._fill_value == True
            assert crust_outer._fill_value == False
            assert crust_inner._fill_value == False

        print("Extract mesh")
        with timed("\tTime: "):
            # Extraction
            mesh_extractor = mesh_extraction.MeshExtraction(mincut)
            vertices, faces = mesh_extractor.extract_mesh()

            ren = VoxelRender()
            fig = ren.make_figure()
            fig.add_trace(ren.make_mesh(vertices, faces, name='Mesh', flatshading=False))
            fig.add_trace(ren.make_wireframe(vertices, faces, name='Wireframe'))
            fig.update_layout(showlegend=True)
            fig.show()

        print("Smoothing mesh")
        with timed("\tTime: "):
            # Smoothing
            pytorch_mesh = pytorch3d.structures.Meshes(verts=[torch.FloatTensor(vertices)],
                                                       faces=[torch.LongTensor(faces)])

            smoothed_vertices = mesh_extraction.Smoothing().smooth(vertices, faces, diff, pytorch_mesh)
            verts = smoothed_vertices.cpu().detach().numpy()
            faces = torch.cat(pytorch_mesh.faces_list()).cpu().detach().numpy()

            ren = VoxelRender()
            fig = ren.make_figure()
            fig.add_trace(ren.make_mesh(verts, faces, name='Mesh', flatshading=False))
            fig.add_trace(ren.make_wireframe(verts, faces, name='Wireframe'))
            fig.update_layout(showlegend=True)
            fig.show()
