import numpy as np
import multiprocessing
from model.model_pts import FixedPtsModels
import enum
from data.chunks import ChunkGrid, ChunkFace
from mathlib import Vec3i, Vec3f
from typing import Tuple, Sequence
from render.cloud_render import CloudRender
from utils import timed
from crust_fix import crust_fix
from render.voxel_render import VoxelRender


class Edge(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    TOP = 2
    BOTTOM = 3
    EAST = 4
    WEST = 5

    def flip(self) -> "Edge":
        return Edge((self // 2) * 2 + ((self + 1) % 2))


class PolyMesh:
    def __init__(self):
        self.vertices = {}
        self.num_verts = 0
        self.faces = []

    def add_vertex(self, v: Vec3f) -> int:
        if tuple(v) in self.vertices:
            return self.vertices[tuple(v)]
        else:
            self.vertices[tuple(v)] = self.num_verts
            self.num_verts += 1
            return self.num_verts - 1

    def get_vertex_array(self) -> np.array:
        vertices = np.zeros((len(self.vertices), 3), dtype=float)
        for v in self.vertices.keys():
            vertices[self.vertices[v]] = np.array(v)
        return vertices


def extract_mesh(segment0: ChunkGrid[np.bool8], segment1: ChunkGrid[np.bool8], segments: np.ndarray):
    s_opt = segment0 & segment1
    block = get_block(get_first_block(s_opt))
    mesh = PolyMesh()
    block_num = 0
    invalid_faces = 0
    while block is not None:
        if not make_face(block, mesh, segments, s_opt):
            invalid_faces += 1
        next_block_pos = get_next_block(block[0], s_opt)
        if next_block_pos is not None:
            block = get_block(next_block_pos)
        else:
            break
        block_num += 1
    print('block num ', block_num)
    print('invalid faces ', invalid_faces)
    return mesh


def make_face(block: np.ndarray, mesh: PolyMesh, segments: np.ndarray, s_opt: ChunkGrid[np.bool8]):
    starting_voxel, current_edge = get_starting_edge(block[0], s_opt, segments)
    current_voxel, next_voxel = starting_voxel, -1
    face = []
    while starting_voxel != next_voxel:
        mesh.add_vertex(block[current_voxel])
        face.append(mesh.add_vertex(block[current_voxel]))
        cut_edges = get_cut_edges(current_voxel, get_nodes(block[current_voxel]), segments)
        next_edge = set(cut_edges) - {current_edge}
        assert next_edge, "No adjacent cut edge found."
        next_edge = next_edge.pop()
        for i, v in enumerate(block):
            if i == current_voxel:
                continue
            nodes = get_nodes(v)
            if nodes is None:
                continue
            edges = get_cut_edges(i, nodes, segments)
            if next_edge in edges:
                next_voxel = i
                current_voxel = next_voxel
                current_edge = next_edge
                break

    if len(face) >= 3:
        mesh.faces.append(face)
        return True
    else:
        return False
    #assert len(face) >= 3, "Face has only " + str(len(face)) + " vertices."
    #mesh.faces.append(face)


def get_starting_edge(start_pos: Vec3i, s_opt: ChunkGrid[np.bool8], segments: np.ndarray) -> (int, tuple):
    block = get_block(start_pos)
    for i, pos in enumerate(block):
        if s_opt.get_value(pos):
            cut_edges = get_cut_edges(i, get_nodes(pos), segments)
            if len(cut_edges) < 2:
                continue
            assert len(cut_edges) == 2, "Cut edges must be 2, but is " + str(len(cut_edges))
            return i, cut_edges[0]
    raise RuntimeError("No starting cut edge found.")


def get_nodes(pos: Vec3i) -> np.ndarray:
    indices = []
    for i in range(6):
        index = get_node(tuple(pos), ChunkFace(i))
        if index in nodes_index:
            indices.append(index)
    if len(indices) == 6:
        return np.array([nodes_index[i] for i in indices])


def get_cut_edges(block_id, nodes: np.ndarray, segments: np.ndarray) -> tuple:
    cut_edges = []
    edges = block_idx_to_edges(block_id)
    for e in edges:
        n = block_edge_to_nodes(block_id, e, nodes)
        if segments[n[0]] != segments[n[1]]:
            cut_edges.append(e)
    assert len(cut_edges) <= 2, "Each voxel should have at most 2 cut edges."
    return tuple(cut_edges)


def get_first_block(s_opt: ChunkGrid[np.bool8]) -> Vec3i:
    for c, chunk in s_opt.chunks.items():
        if not chunk.any():
            continue
        for pos_x, x in enumerate(chunk.to_array()):
            for pos_y, y in enumerate(x):
                for pos_z, z in enumerate(y):
                    pos = np.array((pos_x, pos_y, pos_z)) + np.array(c) * s_opt.chunk_size
                    if has_face(pos, s_opt):
                        return pos
    raise RuntimeError("No starting block can be found.")


def get_next_block(start_pos: Vec3i, s_opt: ChunkGrid[np.bool8]) -> Vec3i:
    inner_pos = np.mod(start_pos, s_opt.chunk_size)
    chunk = s_opt.chunk_at_pos(start_pos)
    if tuple(inner_pos) is not (s_opt.chunk_size - 1, ) * 3:
        if inner_pos[2] < s_opt.chunk_size - 1:
            for pos_z, z in enumerate(chunk.to_array()[inner_pos[0], inner_pos[1], inner_pos[2]+1:]):
                pos = (inner_pos[0], inner_pos[1], pos_z+inner_pos[2]+1) + np.array(chunk.index) * s_opt.chunk_size
                if has_face(pos, s_opt):
                    return pos
        if inner_pos[1] < s_opt.chunk_size - 1:
            for pos_y, y in enumerate(chunk.to_array()[inner_pos[0], inner_pos[1]+1:]):
                for pos_z, z in enumerate(y):
                    pos = (inner_pos[0], pos_y+inner_pos[1]+1, pos_z) + np.array(chunk.index) * s_opt.chunk_size
                    if has_face(pos, s_opt):
                        return pos
        if inner_pos[0] < s_opt.chunk_size - 1:
            for pos_x, x in enumerate(chunk.to_array()[inner_pos[0]+1:]):
                for pos_y, y in enumerate(x):
                    for pos_z, z in enumerate(y):
                        pos = (pos_x+inner_pos[0]+1, pos_y, pos_z) + np.array(chunk.index) * s_opt.chunk_size
                        if has_face(pos, s_opt):
                            return pos
    start_iter = False
    for c, c_ in s_opt.chunks.items():
        if c == tuple(chunk.index):
            start_iter = True
            continue
        elif not start_iter:
            continue
        if not c_.any():
            continue
        for pos_x, x in enumerate(chunk.to_array()):
            for pos_y, y in enumerate(x):
                for pos_z, z in enumerate(y):
                    pos = np.array((pos_x, pos_y, pos_z)) + np.array(c) * s_opt.chunk_size
                    if has_face(pos, s_opt):
                        return pos


def has_face(pos: Vec3i, s_opt: ChunkGrid[np.bool8]):
    if tuple(pos) == (31, 56, 15):
        hello = 'hi'  # this block has 4 voxels with cut edges. voxel 0 and 2 share 2 cut edges and 5 and 7 share 2 cut edges.
    block = get_block(pos)
    count = 0
    for i, p in enumerate(block):
        if not s_opt.get_value(p):
            continue
        nodes = get_nodes(p)
        if nodes is not None and (len(get_cut_edges(i, nodes, segments)) == 2):
            count += 1
    if count >= 3:
        return True
    else:
        return False


def get_block(pos: Vec3i):
    idx = np.array(((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)))
    return idx + pos


def block_idx_to_edges(idx: int) -> Tuple[Edge, Edge, Edge]:
    """

    :param idx: Index of the voxel in the 2x2x2 block. Can be 0 to 7.
    :return: Edges adjacent to the center corner of the block.
    """
    idx_to_edges = {
        0: (Edge.SOUTH, Edge.BOTTOM, Edge.WEST),
        1: (Edge.SOUTH, Edge.BOTTOM, Edge.EAST),
        2: (Edge.SOUTH, Edge.TOP, Edge.WEST),
        3: (Edge.SOUTH, Edge.TOP, Edge.EAST),
        4: (Edge.NORTH, Edge.BOTTOM, Edge.WEST),
        5: (Edge.NORTH, Edge.BOTTOM, Edge.EAST),
        6: (Edge.NORTH, Edge.TOP, Edge.WEST),
        7: (Edge.NORTH, Edge.TOP, Edge.EAST)
    }
    assert idx in idx_to_edges, "Invalid block index."
    return idx_to_edges[idx]


def block_edge_to_nodes(block_idx: int, edge: Edge, nodes: np.ndarray) -> tuple:
    edges = block_idx_to_edges(block_idx)
    edges = set(edges) - {edge}
    return nodes[edges.pop().flip()], nodes[edges.pop().flip()]


def smoothe():
    pass


def make_triangles(mesh: PolyMesh):
    triangles = []
    for f in mesh.faces:
        if len(f) == 3:
            triangles.append(f)
            continue
        for i in range(len(f)-2):
            triangles.append([f[0], f[i+1], f[i + 2]])
    return np.array(triangles)


NodeIndex = Tuple[Vec3i, ChunkFace]


def get_node(pos: Vec3i, face: ChunkFace) -> NodeIndex:
    """Basically forces to have only positive-direction faces"""
    if face % 2 == 0:
        return pos, face
    else:
        return tuple(np.add(pos, face.direction, dtype=int)), face.flip()


def to_voxel(nodeIndex: NodeIndex) -> Sequence[Vec3i]:
    pos, face = nodeIndex
    return [
        pos,
        np.asarray(face.direction) * (face.flip() % 2) + pos
    ]


if __name__ == '__main__':
    CHUNKSIZE = 16
    resolution = 64

    with multiprocessing.Pool() as pool:
        print("Loading model")
        with timed("\tTime: "):
            data = FixedPtsModels.bunny()
            data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
            model: ChunkGrid[np.bool8] = ChunkGrid(CHUNKSIZE, dtype=np.bool8, fill_value=np.bool8(False))
            model[data_pts] = True
            model.pad_chunks(2)
            model.cleanup()

        initial_crust: ChunkGrid[np.bool8] = model.copy()

        """
        Only one dilation to find the outer and inner crust at a lower resolution.
        """
        print("Dilation")
        with timed("\tTime: "):
            crust, components, dilation_step = crust_dilation(initial_crust, max_steps=CHUNKSIZE * 2)
            crust_dilate = dilate(crust)
            outer_fill = components == 2
            crust_outer = outer_fill & crust_dilate
            crust_inner = (components != 1) & (components != 2) & crust_dilate

        """
        Approximate Voxel near Medial Axis, by propagating a Normal field inwards.
        Then for each voxel compute a normal cone and mark the voxel as inner component when the cone angle is greater than 90°.
        """

        for resolution_step in range(0, 1):
            print(f"RESOLUTION STEP: {resolution_step}")

            print("Crust-Fix")
            with timed("\tTime: "):
                crust_inner |= crust_fix(crust, outer_fill, crust_outer, min_distance=dilation_step,
                                         crust_inner=crust_inner, data_pts=data_pts, pool=pool)
                crust_inner[model] = False  # Remove model voxels if they have been added by the crust fix

            print("Diffusion")
            with timed("\tTime: "):
                diff = diffuse(model, repeat=3)

            print("MinCut")
            with timed("\tTime: "):
                segment0, segment1, segments, voxels, nodes_index = mincut(diff, crust, crust_outer, crust_inner)
                thincrust = segment0 & segment1

            print("Dilation")
            with timed("\tTime: "):
                crust, components, dilation_step = crust_dilation(initial_crust)
                crust_outer = dilate(components == 2) & crust
                crust_inner = dilate((components != 1) & (components != 2)) & crust

            print("Volumetric refinment")
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
                crust = dilate(crust, steps=2)

                components, count = fill_components(crust, max_components=2)
                crust_dilate = dilate(crust)
                outer_fill = components == 2
                crust_outer = outer_fill & crust_dilate
                crust_inner = (components != 1) & (components != 2) & crust_dilate

    print("Extract mesh")
    with timed("\tTime: "):
        mesh = extract_mesh(segment0, segment1, segments)
        triangles = make_triangles(mesh)
        ren = VoxelRender()
        fig = ren.make_figure()
        fig.add_trace(ren.make_mesh(mesh.get_vertex_array(), triangles, name='Mesh'))
        fig.show()
