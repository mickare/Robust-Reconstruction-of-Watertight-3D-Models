"""
Mesh extractions from voxels
"""
from typing import Tuple, List, Set, Dict

import numba
import numpy as np
import torch
from pytorch3d.structures import Meshes

from reconstruction.data.chunks import ChunkGrid, ChunkFace
from reconstruction.filters.normals import grid_normals
from reconstruction.mathlib import Vec3i
from reconstruction.mincut import MinCut
from reconstruction.render import voxel_render

numba.config.THREADING_LAYER = 'omp'

NodeIndex = Tuple[Vec3i, ChunkFace]


@numba.stencil(cval=0)
def _sum_block222(a):
    return (a[0, 0, 0] + a[0, 0, 1] + a[0, 1, 0] + a[0, 1, 1] + a[1, 0, 0] + a[1, 0, 1] + a[1, 1, 0] + a[1, 1, 1])


CutEdge = int
CutEdge_X = 1
CutEdge_Y = 2
CutEdge_Z = 4
CutEdge_NONE = 0


@numba.njit()
def _select_starting_voxel(block: np.ndarray):
    res = np.argwhere(block)
    if len(res):
        return res[0][0], res[0][1], res[0][2]
    raise RuntimeError("Block is not valid")


@numba.njit(fastmath=True)
def _detect_cut_edges(block: np.ndarray, face_segments: np.ndarray):
    """
    Extract the cut edges for each voxel
    :param block: 2x2x2 boolean array
    :param face_segments: 2x2x2x6 boolean array
    :return: 2x2x2 bit-mask array of CutEdge
    """
    result = np.zeros((2, 2, 2), dtype=np.int8)
    for x, y, z in np.ndindex((2, 2, 2)):
        if block[x, y, z]:
            fx = ChunkFace.NORTH + x  # Flip when x is 1, works because EnumInt ChunkFace is base 2
            fy = ChunkFace.TOP + y
            fz = ChunkFace.EAST + z
            r = 0
            # Find cut-edges where the segmentation is different
            if face_segments[x, y, z, fx] != face_segments[x, y, z, fy]:
                r |= CutEdge_Z
            if face_segments[x, y, z, fx] != face_segments[x, y, z, fz]:
                r |= CutEdge_Y
            if face_segments[x, y, z, fy] != face_segments[x, y, z, fz]:
                r |= CutEdge_X
            result[x, y, z] = r
    return result


@numba.njit(fastmath=True, inline='always')
def _any_edge(edge: int):
    if edge & CutEdge_X:
        return CutEdge_X
    elif edge & CutEdge_Y:
        return CutEdge_Y
    elif edge & CutEdge_Z:
        return CutEdge_Z
    return 0


@numba.njit(fastmath=True)
def _iter_block_neighbors(pos: Vec3i):
    n = _block_pos_to_idx(pos)
    for i in range(8):
        idx = n + i
        p = _block_idx_to_pos(idx)
        if _is_block_neighbor(pos, p):
            yield p


@numba.njit(fastmath=True, inline='always')
def _is_edge_neighbor(edge: int, a: Vec3i, b: Vec3i):
    idx = edge >> 1
    return a[idx] == b[idx]


@numba.njit(fastmath=True)
def _block_pos_to_idx(pos: Vec3i) -> int:
    return pos[0] | pos[1] << 1 | pos[2] << 2


@numba.njit(fastmath=True)
def _block_idx_to_pos(idx: int) -> Vec3i:
    return (idx & 1, (idx >> 1) & 1, (idx >> 2) & 1)


@numba.njit(fastmath=True)
def _sum_normals(normals: np.ndarray, mask: np.ndarray):
    result = np.zeros((3,), dtype=np.float32)
    for x, y, z in np.argwhere(mask):
        result += normals[x, y, z]
    return result


@numba.njit(fastmath=True)
def _make_face(block: np.ndarray, cut_edges: np.ndarray, normals: np.ndarray):
    """
    Make a face for a 2x2x2 block of the voxels from the mincut graph.
    The face segments must be also a block with each entry with 6 booleans for each face.
    :param block: 2x2x2 boolean array
    :param face_segments: 2x2x2 bit-mask array of CutEdge
    :return:
    """

    sopt_union = block & (cut_edges != 0)  # Union Sopt and B
    if np.sum(sopt_union) < 3:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.uint32)

    # Collect polygon edges
    pos0 = _select_starting_voxel(sopt_union)
    edge0 = _any_edge(cut_edges[pos0])

    if edge0 == CutEdge_NONE:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.uint32)

    pos = pos0
    edge = edge0
    visited: List[Vec3i] = [pos]
    while True:
        edge1 = _any_edge(cut_edges[pos] & ~edge)  # Find the second cut edge f in v
        assert edge1 != CutEdge_NONE
        found = False
        for repeat in range(2):
            for pos1 in _iter_block_neighbors(pos):  # Find the neigboring voxel w
                if repeat == 1 or pos1 not in visited:
                    if _is_edge_neighbor(edge1, pos, pos1):
                        if block[pos1] and cut_edges[pos1] & edge1:  # that shares the cut-edge
                            visited.append(pos)  # generate a polygon edge from v to w
                            pos = pos1  # v <- w
                            edge = edge1  # e <- f
                            found = True
                            break
            if found:
                break

        if not found or pos == pos0:  # loop until first voxel is reached again
            break

    len_visited = len(visited)
    if len_visited <= 2:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.uint32)

    # Normal approximation of this block
    # normal = _sum_normals(normals, sopt_union)
    # normal_len = np.linalg.norm(normal)
    # min_cos_angle = 0

    # Construct vertices and faces
    verts = np.array(visited, dtype=np.int32)
    faces = np.zeros((len_visited - 2, 3), dtype=np.uint32)
    for i in range(len_visited - 2):
        f1 = i + 1
        f2 = i + 2
        vertNormal = np.cross(verts[f1] - verts[0], verts[f2] - verts[0]).astype(np.float32)
        approxNormal = (normals[visited[0]] + normals[visited[f1]] + normals[visited[f2]])

        # Swap if angle (scalar product) is greater than 90°.
        if (vertNormal @ approxNormal) < 0:  # cos(90° degree max angle) * |vertNormal| * |approxNormal|
            faces[i] = (0, f2, f1)  # swap vertices to rotate face
        else:
            faces[i] = (0, f1, f2)
    return verts, faces


@numba.njit(fastmath=True)
def _extract_polygon_edges(sopt: np.ndarray, segments: np.ndarray, normals: np.ndarray) \
        -> List[Tuple[np.ndarray, np.ndarray]]:
    detected = _sum_block222(sopt)
    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for pos in np.argwhere(detected >= 3):
        x, y, z = pos
        sopt_block = sopt[x:x + 2, y:y + 2, z:z + 2]
        cut_edges = _detect_cut_edges(sopt_block, segments[x:x + 2, y:y + 2, z:z + 2])
        normals_block = normals[x:x + 2, y:y + 2, z:z + 2]
        v, f = _make_face(sopt_block, cut_edges, normals_block)
        if len(v) > 0:
            result.append((v + pos, f))

    # Fallback, empty result
    return result


def extract_voxel_segments(nodes_index: Dict[NodeIndex, int], segments: np.ndarray, sopt: np.ndarray, offset: Vec3i):
    """Compute segment for each voxel in sopt"""
    _get_node = MinCut.get_node
    start = np.asanyarray(offset)
    result = np.zeros((*sopt.shape, 6), dtype=np.bool8)
    for p in np.argwhere(sopt):
        pos = start + p
        tpos = tuple(pos)
        value = [segments[nodes_index[_get_node(tpos, f)]] for f in ChunkFace]
        assert any(value)  # Make sure that the segment is valid!
        result[p[0], p[1], p[2], :] = value
    return result


class MeshExtraction:
    def __init__(self, mincut: MinCut):
        self.segment0, self.segment1 = mincut.grid_segments()
        self.s_opt = self.segment0 & self.segment1
        self.mincut = mincut

    def extract_mesh(self):
        sopt = self.s_opt
        # Clean chunk padding
        sopt.cleanup(remove=True)
        sopt.pad_chunks(1)

        size = sopt.chunk_size

        # Cache mincut segments
        segments = self.mincut.segments()
        nodes_index = self.mincut.nodes_index

        # Approximate surface normals via outer segment
        normals = grid_normals(sopt, outer=self.segment0)

        result: List[Tuple[np.ndarray, np.ndarray]] = []
        for index in sopt.chunks.keys():
            position = np.asanyarray(index, dtype=np.int32) * size
            block = sopt.get_block_at(index, (2, 2, 2), offset=(0, 0, 0))
            pad = sopt.block_to_array(block)
            block_sopt = pad[:size + 1, :size + 1, :size + 1]

            # Surface normals
            arr_normals = normals.block_to_array(normals.get_block_at(index, (2, 2, 2), offset=(0, 0, 0)))
            block_normals = arr_normals[:size + 1, :size + 1, :size + 1]

            # Octaeder segments
            voxel_segments = extract_voxel_segments(nodes_index, segments, block_sopt, position)

            v, f = voxel_render.reduce_mesh(_extract_polygon_edges(block_sopt, voxel_segments, block_normals))
            # v, f = voxel_render.reduce_mesh(_extract_polygon_edges(block_sopt))
            v += position
            result.append((v, f))

        # unpack results
        vs, fs = voxel_render.reduce_mesh(result)
        return vs.astype(dtype=np.float32) + 0.5, fs.astype(np.int32)


class Smoothing:
    def smooth(self, vertices: np.ndarray, faces: np.ndarray, diffusion: ChunkGrid[float], original_mesh: Meshes,
               max_iteration=50):
        assert max_iteration > -1
        change = True
        iteration = 0
        loss_mesh = original_mesh.clone()
        smooth_verts = loss_mesh.verts_packed().clone()
        neighbors = self.compute_neighbors(vertices, faces)
        neighbor_len = torch.IntTensor([len(neighbors[i]) for i in range(len(vertices))])
        neighbor_valences = torch.FloatTensor(
            [sum([1 / neighbor_len[n] for n in neighbors[i]]) for i in range(len(vertices))])
        d = 1 + 1 / neighbor_len * neighbor_valences

        difference_max = torch.as_tensor(diffusion.get_values(vertices) + 1)

        while change and iteration < max_iteration:
            iteration += 1
            change = False

            for i in range(2):
                with torch.no_grad():
                    L = loss_mesh.laplacian_packed()
                loss = L.mm(loss_mesh.verts_packed())
                if i == 0:
                    loss_mesh = Meshes([loss], loss_mesh.faces_list())

            # new_vals = smooth_verts - (1/d).unsqueeze(1) * loss
            # difference = torch.sqrt(torch.sum(torch.pow(original_mesh.verts_packed() - new_vals, 2), dim=1))

            new_val = smooth_verts - (loss.T * (1 / d)).T
            differences = torch.linalg.norm(original_mesh.verts_packed() - new_val, dim=1)
            cond = differences < difference_max
            if torch.any(cond):
                smooth_verts[cond] = new_val[cond]
                change = True

            # for i, v in enumerate(vertices):
            #     new_val = smooth_verts[i] - (1 / d[i] * loss[i])
            #     difference = torch.dist(original_mesh.verts_packed()[i], new_val)
            #     if difference < difference_max[i]:
            #         smooth_verts[i] = new_val
            #         change = True

            loss_mesh = Meshes([smooth_verts], original_mesh.faces_list())
        return smooth_verts

    def compute_neighbors(self, vertices: np.ndarray, faces: np.ndarray) -> List[Set]:
        neighbors = [set() for i in vertices]
        for face in faces:
            for v in face:
                neighbors[v].update(face)
        for v, s in enumerate(neighbors):
            s.remove(v)
        return neighbors


@numba.njit(fastmath=True, inline='always')
def _distance_block(a: Vec3i, b: Vec3i):
    return abs(b[0] - a[0]) + abs(b[1] - a[1]) + abs(b[2] - a[2])


@numba.njit(fastmath=True, inline='always')
def _is_block_neighbor(a: Vec3i, b: Vec3i):
    s = _distance_block(a, b)
    return s == 1 or s == 2  # Direct neighbor or edge neighbor; but not identity and not diagonal!
