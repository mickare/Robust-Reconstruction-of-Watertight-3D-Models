from typing import Optional, Tuple, Sequence

import maxflow
import numpy as np
import tqdm
from scipy import ndimage

from crust_fix import crust_fix
from data.chunks import ChunkGrid, Chunk
from data.faces import ChunkFace
from filters.dilate import dilate
from filters.fill import flood_fill_at
from mathlib import Vec3i, Vec3f
import mesh_extraction
from model.model_pts import FixedPtsModels
from render.cloud_render import CloudRender
from render.voxel_render import VoxelRender
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
    fig.add_trace(ren.grid_voxel(components > 2, opacity=1.0, name=f"More"))
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


def crust_dilation(crust: ChunkGrid[np.bool8], max_components=4, min_steps=3, max_steps=10):
    assert max_steps > 0
    max_count = 0
    dilation_step = 0
    crusts_all = []
    components_all = []

    for dilation_step in range(max_steps):
        print(f"\t\tDilation-Step {dilation_step}")
        components, count = fill_components(crust, max_components=max_components)
        crusts_all.append(crust)
        components_all.append(components)

        # plot_voxels(components == 0, components)
        # print(count)

        if dilation_step >= min_steps and max_count >= count and count <= 3:
            break
        else:
            max_count = max(max_count, count)
            crust = dilate(crust)
            assert crust.any()

    print("\tSteps: ", dilation_step)

    # Take the crust one step before the inner component vanished.
    crust = crusts_all[max(0, len(crusts_all) - 2)]
    components = components_all[max(0, len(components_all) - 2)]
    # crust = crusts_all[-1]
    # components = components_all[-1]
    crust.cleanup(remove=True)
    components.cleanup(remove=True)
    return crust, components, dilation_step


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


# =====================================================================
# MinCut
# =====================================================================


def mincut(diff: ChunkGrid[float], crust: ChunkGrid[bool], crust_outer: ChunkGrid[bool],
           crust_inner: ChunkGrid[bool], s=4, a=1e-20):
    # weights = ChunkGrid(diff.chunk_size, dtype=float, fill_value=0)
    weights = (diff ** s) + a
    weights[~crust] = -1
    weights.cleanup(remove=True)

    NodeIndex = Tuple[Vec3i, ChunkFace]
    CF = ChunkFace

    def get_node(pos: Vec3i, face: ChunkFace) -> NodeIndex:
        """Basically forces to have only positive-direction faces"""
        if face % 2 == 0:
            return pos, face
        else:
            return tuple(np.add(pos, face.direction(), dtype=int)), face.flip()

    voxels = {tuple(p): w for p, w in weights.items(mask=crust) if w >= 0}
    nodes = list(set(get_node(p, f) for p in voxels.keys() for f in ChunkFace))
    nodes_index = {f: n for n, f in enumerate(nodes)}

    nodes_count = len(nodes)

    graph = maxflow.Graph[float](nodes_count, nodes_count)
    g_nodes = graph.add_nodes(len(nodes))

    # visited: Set[Tuple[Tuple[Vec3i, CF], Tuple[Vec3i, CF]]] = set()
    for vPos, w in tqdm.tqdm(voxels.items(), total=len(voxels), desc="Linking Faces"):
        iN = nodes_index[get_node(vPos, CF.NORTH)]
        iS = nodes_index[get_node(vPos, CF.SOUTH)]
        iT = nodes_index[get_node(vPos, CF.TOP)]
        iB = nodes_index[get_node(vPos, CF.BOTTOM)]
        iE = nodes_index[get_node(vPos, CF.EAST)]
        iW = nodes_index[get_node(vPos, CF.WEST)]
        for f, o in [
            (iN, iE), (iN, iW), (iN, iT), (iN, iB),
            (iS, iE), (iS, iW), (iS, iT), (iS, iB),
            (iT, iE), (iT, iW), (iB, iE), (iB, iW)
        ]:  # type: ChunkFace
            graph.add_edge(f, o, w, w)

    # Source
    for vPos in tqdm.tqdm(list(crust_outer.where()), desc="Linking Source"):
        for f in ChunkFace:  # type: ChunkFace
            fNode = get_node(tuple(vPos), f)
            fIndex = nodes_index.get(fNode, None)
            if fIndex is not None:
                graph.add_tedge(fIndex, 10000, 0)

    # Sink
    for vPos in tqdm.tqdm(list(crust_inner.where()), desc="Linking Sink"):
        for f in ChunkFace:  # type: ChunkFace
            fNode = get_node(tuple(vPos), f)
            fIndex = nodes_index.get(fNode, None)
            if fIndex is not None:
                graph.add_tedge(fIndex, 0, 10000)

    flow = graph.maxflow()
    segments = graph.get_grid_segments(np.arange(nodes_count))

    def to_voxel(nodeIndex: NodeIndex) -> Sequence[Vec3i]:
        pos, face = nodeIndex
        return [
            pos,
            np.asarray(face.direction()) * (face.flip() % 2) + pos
        ]

    segment0 = ChunkGrid(crust.chunk_size, bool, False)
    segment0[[p for node, s in zip(nodes, segments) if s == False
              for p in to_voxel(node)]] = True
    segment1 = ChunkGrid(crust.chunk_size, bool, False)
    segment1[[p for node, s in zip(nodes, segments) if s == True
              for p in to_voxel(node)]] = True

    return segment0, segment1, segments, voxels, nodes_index


# =====================================================================
# Render
# =====================================================================
if __name__ == '__main__':
    CHUNKSIZE = 16
    resolution = 64

    print("Loading model")
    with timed("\tTime: "):
        data = FixedPtsModels.bunny()
        # data = PtsModelLoader().load("models/bunny/bunnyData.pts")
        data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
        model: ChunkGrid[np.bool8] = ChunkGrid(CHUNKSIZE, dtype=np.bool8, fill_value=np.bool8(False))
        model[data_pts] = True
        model.pad_chunks(2)
        model.cleanup()

    initial_crust: ChunkGrid[np.bool8] = model.copy()
    initial_crust.cleanup(remove=True)

    # ren = VoxelRender()
    # fig = ren.make_figure()
    # fig.add_trace(ren.grid_voxel(initial_crust, opacity=0.1, name='Initial'))
    # fig.add_trace(CloudRender().make_scatter(data_pts, size=1, name='Model'))
    # fig.show()

    """
    Only one dilation to find the outer and inner crust at a lower resolution.
    """
    print("Dilation")
    with timed("\tTime: "):
        crust, components, dilation_step = crust_dilation(initial_crust, max_steps=CHUNKSIZE * 2)
        # assert components._fill_value == 2

        plot_voxels(components == 0, components, title=f"Initial Dilation")
        crust_dilate = dilate(crust)
        outer_fill = components == 2
        crust_outer = outer_fill & crust_dilate
        crust_inner = (components != 1) & (components != 2) & crust_dilate

        assert crust_dilate._fill_value == False
        assert outer_fill._fill_value == True
        assert crust_outer._fill_value == False
        assert crust_inner._fill_value == False

    """
    Increase resolution and make the mesh approximation finer
    """
    for resolution_step in range(0, 1):
        print(f"RESOLUTION STEP: {resolution_step}")

        """
        Approximate Voxel near Medial Axis, by propagating a Normal field inwards.
        Then for each voxel compute a normal cone and mark the voxel as inner component when the cone angle is greater than 90°.
        """
        print("Crust-Fix")
        with timed("\tTime: "):
            crust_inner |= crust_fix(crust, outer_fill, crust_outer, crust_inner,
                                     min_distance=dilation_step, data_pts=data_pts)
            # crust_inner[model] = False  # Remove model voxels if they have been added by the crust fix

        print("Render Crust")
        with timed("\tTime: "):
            ren = VoxelRender()
            fig = ren.make_figure(title=f"Step-{resolution_step}: Crust")
            fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.1, name='Outer'))
            fig.add_trace(ren.grid_voxel(crust_inner, opacity=1.0, name='Inner'))
            fig.add_trace(CloudRender().make_scatter(data_pts, size=0.7, name='Model'))
            fig.show()

        print("Diffusion")
        with timed("\tTime: "):
            diff = diffuse(model, repeat=3)

        print("MinCut")
        with timed("\tTime: "):
            segment0, segment1, segments, voxels, nodes_index = mincut(diff, crust, crust_outer, crust_inner)
            thincrust = segment0 & segment1

        print("Render")
        with timed("\tTime: "):
            ren = VoxelRender()
            fig = ren.make_figure(title=f"Step-{resolution_step}: Segments")
            fig.add_trace(ren.grid_voxel(segment0, opacity=0.1, name='Segment 0'))
            fig.add_trace(ren.grid_voxel(segment1, opacity=0.1, name='Segment 1'))
            fig.add_trace(ren.grid_voxel(thincrust, opacity=1.0, name='Join'))
            fig.add_trace(CloudRender().make_scatter(data_pts, size=1, name='Model'))
            fig.show()

        print("Volumetric refinement")
        with timed("\tTime: "):
            # Rebuild model
            resolution *= 2
            data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
            model = ChunkGrid(initial_crust.chunk_size, np.bool8, fill_value=np.bool8(False))
            model[data_pts] = np.bool8(True)
            model.pad_chunks(1)
            model.cleanup()

            # Build new crust
            crust = thincrust.split(2) | model
            crust = dilate(crust, steps=1)

            components, count = fill_components(crust, max_components=2)
            outer_fill = components == 2

            # Make a clean padding
            outer_fill.cleanup(remove=True)
            outer_fill.pad_chunks(1)

            crust_dilate = dilate(crust)
            crust_outer = outer_fill & crust_dilate
            crust_inner = (components != 1) & (components != 2) & crust_dilate

            # Validate data
            assert crust._fill_value == False
            assert outer_fill._fill_value == True
            assert crust_outer._fill_value == False
            assert crust_inner._fill_value == False

        print("Extract mesh")
        with timed("\tTime: "):
            mesh_extractor = mesh_extraction.MeshExtraction(segment0, segment1, segments, nodes_index)
            mesh = mesh_extractor.extract_mesh()
            triangles = mesh_extractor.make_triangles()
            ren = VoxelRender()
            fig = ren.make_figure()
            fig.add_trace(ren.make_mesh(mesh.get_vertex_array(), triangles, name='Mesh'))
            fig.show()
