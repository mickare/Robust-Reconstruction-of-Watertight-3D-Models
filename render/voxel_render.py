from typing import Tuple, Sequence, Iterable, Optional, List

import numba
import numpy as np
import plotly.graph_objects as go
import tqdm

from data.chunks import Chunk, ChunkGrid, ChunkFace
from mathlib import Vec3f
from utils import merge_default


def immutable(arr: np.ndarray):
    arr.flags.writeable = False
    return arr


def _transfer_face(face: ChunkFace, dst: np.ndarray, neighbor: Optional[Chunk]):
    if neighbor is not None:
        if neighbor.is_filled():
            dst[face.slice(other=slice(1, -1))] = np.bool8(neighbor.value)
        else:
            dst[face.slice(other=slice(1, -1))] = neighbor.to_array()[face.flip().slice()].astype(dtype=np.bool8)


@numba.njit(inline='always')
def _empty_faces(vtype=np.int32) -> Tuple[np.ndarray, np.ndarray]:
    return np.empty((0, 3), dtype=vtype), np.empty((0, 3), dtype=np.uint32)


def reduce_mesh(vertices_faces: Sequence[Tuple[np.ndarray, np.ndarray]], vtype=np.int32) \
        -> Tuple[np.ndarray, np.ndarray]:
    # Check if empty
    if len(vertices_faces) == 0:
        return _empty_faces(vtype=vtype)
    # Increment face indices and filter empty
    vs = []
    fs = []
    face_index = 0
    for v, f in vertices_faces:
        if len(v) == 0 or len(f) == 0:
            continue
        vs.append(v)
        fs.append(f + face_index)
        face_index += len(v)
    # Re-check if empty
    if len(vs) == 0 or len(fs) == 0:
        return _empty_faces(vtype=vtype)
    # Stack vertices and faces
    vs2 = np.vstack(vs)
    fs2 = np.vstack(fs)
    # Remove duplicates
    vs3, inv = np.unique(vs2, return_inverse=True, axis=0)
    fs3 = inv[fs2]
    return vs3, fs3


@numba.njit(parallel=True, fastmath=True)
def _make_faces_from_delta(delta: np.ndarray, vert: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _faces_front = np.array([(0, 1, 2), (3, 2, 1)], dtype=np.uint32)
    _faces_back = np.array([(3, 2, 1), (0, 1, 2)], dtype=np.uint32)

    di = np.argwhere(delta != 0)
    if not np.any(di):
        return _empty_faces()

    # Faces
    f = np.empty((len(di), 1, 1), dtype=np.uint32)
    for i in numba.prange(len(di)):
        f[i] = _faces_back if (delta[di[i]] < 0) else _faces_front

    # Flatten faces
    faces = (f.T + 4 * np.arange(len(di))).T.reshape((-1, 3))
    # Vertices
    vertices = (di[:, None] + vert).reshape(-1, 3)

    return vertices, faces


# @numba.njit(parallel=True, fastmath=True)
# def _make_mesh_from_deltas(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray):
#     _vert = np.array([
#         [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)],  # x
#         [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)],  # y
#         [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],  # z
#     ], dtype=np.int32)
#     _faces_front = np.array([(0, 1, 2), (3, 2, 1)], dtype=np.uint32)
#     _faces_back = np.array([(3, 2, 1), (0, 1, 2)], dtype=np.uint32)
#
#     deltas = (dx, dy, dz)
#     result: List[Tuple[np.ndarray, np.ndarray]] = [_make_faces_from_delta(deltas[n], _vert[n]) for n in numba.prange(3)]
#
#     return _reduce_mesh(result)


def _clip_face(face0: ChunkFace, dst: np.ndarray):
    s0 = face0.slice()
    s1 = face0.flip().slice()
    dst[s0] = np.where(dst[s0] > 0, 0, dst[s0])
    dst[s1] = np.where(dst[s1] < 0, 0, dst[s1])


@numba.njit(parallel=True, fastmath=True)
def _direction_dx(voxels: np.ndarray):
    res = voxels[1:, 1:-1, 1:-1] - voxels[:-1, 1:-1, 1:-1]
    # Remove duplicate faces at border, show only faces that point away
    res[0, :, :] = (res[0, :, :] == 1)
    res[-1, :, :] = (res[-1, :, :] == -1) * -1
    return res


@numba.njit(parallel=True, fastmath=True)
def _direction_dy(voxels: np.ndarray):
    res = voxels[1:-1, 1:, 1:-1] - voxels[1:-1, :-1, 1:-1]
    # Remove duplicate faces at border, show only faces that point away
    res[:, 0, :] = (res[:, 0, :] == 1)
    res[:, -1, :] = (res[:, -1, :] == -1) * -1
    return res


@numba.njit(parallel=True, fastmath=True)
def _direction_dz(voxels: np.ndarray):
    res = voxels[1:-1, 1:-1, 1:] - voxels[1:-1, 1:-1, :-1, ]
    # Remove duplicate faces at border, show only faces that point away
    res[:, :, 0, ] = (res[:, :, 0] == 1)
    res[:, :, -1] = (res[:, :, -1] == -1) * -1
    return res


@numba.njit(parallel=True, fastmath=True)
def _collapse_dx(deltas: np.ndarray):
    col = np.zeros(deltas.shape, dtype=deltas.dtype)
    for yz in numba.pndindex((deltas.shape[1], deltas.shape[2])):
        y, z = yz
        x0 = 0
        value0 = np.nan
        d0 = 1 if value0 > 0 else -1
        for x in range(deltas.shape[0]):
            current = deltas[x, y, z]
            if current and value0 == current:
                col[x0, y, z] += d0
                continue
            x0 = x
            value0 = current
            d0 = 1 if value0 > 0 else -1
            if current:
                col[x0, y, z] = d0
    return col


@numba.njit(parallel=True, fastmath=True)
def _collapse_dy(deltas: np.ndarray):
    col = np.zeros(deltas.shape, dtype=deltas.dtype)
    for xz in numba.pndindex((deltas.shape[0], deltas.shape[2])):
        x, z = xz
        y0 = 0
        value0 = np.nan
        d0 = 1 if value0 > 0 else -1
        for y in range(deltas.shape[1]):
            current = deltas[x, y, z]
            if current and value0 == current:
                col[x, y0, z] += d0
                continue
            y0 = y
            value0 = current
            d0 = 1 if value0 > 0 else -1
            if current:
                col[x, y0, z] = d0
    return col


@numba.njit(parallel=True, fastmath=True)
def _collapse_dz(deltas: np.ndarray):
    col = np.zeros(deltas.shape, dtype=deltas.dtype)
    for xy in numba.pndindex((deltas.shape[0], deltas.shape[1])):
        x, y = xy
        z0 = 0
        value0 = np.nan
        d0 = 1 if value0 > 0 else -1
        for z in range(deltas.shape[2]):
            current = deltas[x, y, z]
            if current and value0 == current:
                col[x, y, z0] += d0
                continue
            z0 = z
            value0 = current
            d0 = 1 if value0 > 0 else -1
            if current:
                col[x, y, z0] = d0
    return col


@numba.njit()
def _merge_dx(dx: np.ndarray):
    dxy = _collapse_dy(dx)
    dxyz = _collapse_dz(dxy)
    return dxy, dxyz


@numba.njit()
def _merge_dy(dy: np.ndarray):
    dxy = _collapse_dx(dy)
    dxyz = _collapse_dz(dxy)
    return dxy, dxyz


@numba.njit()
def _merge_dz(dz: np.ndarray):
    dxz = _collapse_dx(dz)
    dxyz = _collapse_dy(dxz)
    return dxz, dxyz


class MeshHelper:
    _vert_x = immutable(np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)], dtype=np.int32))
    _vert_y = immutable(np.array([(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)], dtype=np.int32))
    _vert_z = immutable(np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)], dtype=np.int32))
    _faces_front = immutable(np.array([(0, 1, 2), (3, 2, 1)], dtype=np.uint32))
    _faces_back = immutable(np.array([(3, 2, 1), (0, 1, 2)], dtype=np.uint32))

    @classmethod
    def _empty(cls) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.uint32)

    @classmethod
    def _reduce_zip(cls, a: Sequence[Sequence], b: Sequence[Sequence]) -> Tuple[Tuple, Tuple]:
        res = tuple(zip(*((u, v) for u, v in zip(a, b) if len(u) != 0 and len(v) != 0)))
        if res:
            a, b = res
            return a, b
        else:
            return (), ()

    @classmethod
    def reduce_mesh(cls, vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not vertices or not faces:
            return cls._empty()
        vertices, faces = cls._reduce_zip(vertices, faces)
        if not vertices or not faces:
            return cls._empty()
        vs = np.vstack(vertices)
        fs = np.vstack(list(cls._join_faces(faces, vertices)))
        vs2, inv = np.unique(vs, return_inverse=True, axis=0)
        fs2 = inv[fs]
        return vs2, fs2

    @classmethod
    def _join_faces(cls, faces: Sequence[np.ndarray], vertices: Sequence[np.ndarray]) -> Iterable[np.ndarray]:
        index = 0
        for f, v in zip(faces, vertices):
            yield f + index
            index += len(v)

    @classmethod
    def extract_voxel_mesh(cls, mask: np.ndarray, neighbors: Sequence[Optional[Chunk]] = None):
        if neighbors is None:
            neighbors = [None] * 6

        u, v, w = mask.shape
        voxels = np.pad(mask.astype(dtype=np.int8), 1, constant_values=0)

        # X-Neighbors
        _transfer_face(ChunkFace.NORTH, voxels, neighbors[ChunkFace.NORTH])
        _transfer_face(ChunkFace.SOUTH, voxels, neighbors[ChunkFace.SOUTH])
        # Y-Neighbors
        _transfer_face(ChunkFace.TOP, voxels, neighbors[ChunkFace.TOP])
        _transfer_face(ChunkFace.BOTTOM, voxels, neighbors[ChunkFace.BOTTOM])
        # Z-Neighbors
        _transfer_face(ChunkFace.EAST, voxels, neighbors[ChunkFace.EAST])
        _transfer_face(ChunkFace.WEST, voxels, neighbors[ChunkFace.WEST])

        # Directions of faces -1 and +1, no face is zero
        dx = _direction_dx(voxels)
        dy = _direction_dy(voxels)
        dz = _direction_dz(voxels)

        # Merge and collapse neighboring voxel faces
        dx_y, dx_z = _merge_dx(dx)
        dy_x, dy_z = _merge_dy(dy)
        dz_x, dz_y = _merge_dz(dz)

        # Find start positions of faces
        ix = np.argwhere(dx_y != 0)
        iy = np.argwhere(dy_x != 0)
        iz = np.argwhere(dz_x != 0)

        # Index vectors of start positions
        tix = tuple(ix.T)
        tiy = tuple(iy.T)
        tiz = tuple(iz.T)

        # Construct scaling of X-Faces
        dix_y = dx_y[tix]
        dix_z = dx_z[tix]
        dix = np.abs(np.transpose((np.ones(len(dix_y)), dix_y, dix_z)))

        # Construct scaling of Y-Faces
        diy_x = dy_x[tiy]
        diy_z = dy_z[tiy]
        diy = np.abs(np.transpose((diy_x, np.ones(len(diy_x)), diy_z)))

        # Construct scaling of Z-Faces
        diz_x = dz_x[tiz]
        diz_y = dz_y[tiz]
        diz = np.abs(np.transpose((diz_x, diz_y, np.ones(len(diz_x)))))

        # Local variable cache
        faces_front = cls._faces_front
        faces_back = cls._faces_back
        vert_x = cls._vert_x
        vert_y = cls._vert_y
        vert_z = cls._vert_z

        # Construct Vertices and Faces, facing X
        vx = (ix[:, None] + (dix[:, None] * vert_x)).reshape((-1, 3))
        fx = np.full((len(ix), 2, 3), faces_front)
        fx[dix_y < 0] = faces_back
        fx = (fx.T + 4 * np.arange(len(ix))).T.reshape((-1, 3))

        # Construct Vertices and Faces, facing Y
        vy = (iy[:, None] + (diy[:, None] * vert_y)).reshape((-1, 3))
        fy = np.full((len(iy), 2, 3), faces_front)
        fy[diy_x < 0] = faces_back
        fy = (fy.T + 4 * np.arange(len(iy))).T.reshape((-1, 3))

        # Construct Vertices and Faces, facing Z
        vz = (iz[:, None] + (diz[:, None] * vert_z)).reshape((-1, 3))
        fz = np.full((len(iz), 2, 3), faces_front)
        fz[diz_x < 0] = faces_back
        fz = (fz.T + 4 * np.arange(len(iz))).T.reshape((-1, 3))

        return cls.reduce_mesh((vx, vy, vz), (fx, fy, fz))

    @classmethod
    def _create_mesh_from_deltas(cls, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray):
        ix = np.argwhere(dx != 0)
        iy = np.argwhere(dy != 0)
        iz = np.argwhere(dz != 0)

        # Local variable cache
        faces_front = cls._faces_front
        faces_back = cls._faces_back
        vert_x = cls._vert_x
        vert_y = cls._vert_y
        vert_z = cls._vert_z

        # Fast in numpy
        vx = (ix[:, None] + vert_x).reshape(-1, 3)
        fx = np.full((len(ix), 2, 3), faces_front)
        fx[dx[tuple(ix.T)] < 0] = faces_back
        fx = (fx.T + 4 * np.arange(len(ix))).T.reshape((-1, 3))

        vy = (iy[:, None] + vert_y).reshape(-1, 3)
        fy = np.full((len(iy), 2, 3), faces_front)
        fy[dy[tuple(iy.T)] < 0] = faces_back
        fy = (fy.T + 4 * np.arange(len(iy))).T.reshape((-1, 3))

        vz = (iz[:, None] + vert_z).reshape(-1, 3)
        fz = np.full((len(iz), 2, 3), faces_front)
        fz[dz[tuple(iz.T)] < 0] = faces_back
        fz = (fz.T + 4 * np.arange(len(iz))).T.reshape((-1, 3))

        return cls.reduce_mesh((vx, vy, vz), (fx, fy, fz))

    @classmethod
    def chunk_to_voxel_mesh(cls, chunk: Chunk, parent: Optional[ChunkGrid] = None, chunked=False) -> Tuple[
        np.ndarray, np.ndarray]:

        if chunked and chunk.is_filled():
            if chunk.value:
                vertices = np.array([
                    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
                ], dtype=np.int32) * chunk.size
                faces = np.array([
                    (0, 1, 2), (3, 2, 1),  # Face
                    (2, 3, 6), (7, 6, 3),  #
                    (3, 1, 7), (5, 7, 1),  #
                    (7, 5, 6), (4, 6, 5),  #
                    (5, 1, 4), (0, 4, 1),  #
                    (4, 0, 6), (2, 6, 0)
                ], dtype=np.uint32)
                return vertices + chunk.index * chunk.size, faces
        elif chunk.any():
            neighbors: List[Optional[Chunk]] = [None] * 6
            if parent is not None:
                neighbors = [c for f, c in parent.iter_neighbors(chunk.index, flatten=False)]
                assert len(neighbors) == 6

            vertices, faces = cls.extract_voxel_mesh(chunk.to_array(), neighbors=neighbors)
            return vertices + chunk.index * chunk.size, faces
        return cls._empty()

    @classmethod
    def grid_to_voxel_mesh(cls, grid: ChunkGrid, verbose=False, name: Optional[str] = None, **kwargs):
        if verbose:
            desc = "Building voxel mesh"
            if name:
                desc = f"Building {name} mesh"
            chunks = tqdm.tqdm(grid.chunks, desc=desc)
        else:
            chunks = grid.chunks
        if grid.chunks:
            __cls_chunk_to_voxel_mesh = cls.chunk_to_voxel_mesh  # Method cache
            vertices, faces = zip(*(__cls_chunk_to_voxel_mesh(c, parent=grid, **kwargs) for c in chunks))
            return cls.reduce_mesh(vertices, faces)
        else:  # no chunks
            return cls._empty()


class VoxelRender:

    def __init__(self):
        self.default_mesh_kwargs = dict(
            lighting=dict(
                ambient=0.18,
                diffuse=1,
                fresnel=0.1,
                specular=0.1,
                roughness=0.05,
                facenormalsepsilon=1e-15,
                vertexnormalsepsilon=1e-15
            ),
            lightposition=dict(x=-1000, y=0, z=300),
            flatshading=True,
        )

    def dense_voxel(self, dense: np.ndarray, **kwargs):
        vertices, faces = MeshHelper.extract_voxel_mesh(dense)
        return self.make_mesh(vertices, faces, **kwargs)

    def grid_voxel(self, grid: ChunkGrid, verbose=False, chunked=False, name: Optional[str] = None, **kwargs):
        vertices, faces = MeshHelper.grid_to_voxel_mesh(grid, verbose=verbose, chunked=chunked, name=name)
        return self.make_mesh(vertices, faces, name=name, **kwargs)

    def grid_wireframe(self, grid: ChunkGrid, verbose=False, chunked=False, name: Optional[str] = None, **kwargs):
        vertices, faces = MeshHelper.grid_to_voxel_mesh(grid, verbose=verbose, chunked=chunked, name=name)
        return self.make_wireframe(vertices, faces, name=name, **kwargs)

    def make_wireframe(self, vertices: np.ndarray, faces: np.ndarray, size=0.5, **kwargs):
        merge_default(kwargs, mode='lines', marker=dict(size=size))
        # arr = np.array([
        #     [
        #         (vertices[min(fi, fj)], vertices[max(fi, fj)]),
        #         (vertices[min(fi, fk)], vertices[max(fi, fk)]),
        #         (vertices[min(fj, fk)], vertices[max(fj, fk)])
        #     ]
        #     for fi, fj, fk in faces
        # ]).reshape((-1, 2, 3))
        # nan = np.ones(3, dtype=vertices.dtype) * np.nan
        # lines = np.array([(l0, l1, nan) for l0, l1 in np.unique(arr, axis=0)]).reshape((-1, 3))

        nan = np.ones(3, dtype=vertices.dtype) * np.nan
        lines = np.array([
            (vertices[fi], vertices[fj], vertices[fk], vertices[fi], nan) for fi, fj, fk in faces
        ]).reshape((-1, 3))
        x, y, z = lines.T
        return go.Scatter3d(x=x, y=y, z=z, **kwargs)

    def make_mesh(self, vertices: np.ndarray, faces: np.ndarray,
                  offset: Optional[Vec3f] = None, **kwargs):
        merge_default(kwargs, **self.default_mesh_kwargs)
        kwargs.setdefault("flatshading", True)

        offset = (0, 0, 0) if offset is None else offset

        vertices = vertices + offset
        x, y, z = self._unwrap(vertices)
        i, j, k = self._unwrap(faces)

        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)

    @classmethod
    def _unwrap(cls, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(data) == 0:
            return np.empty(0), np.empty(0), np.empty(0)
        x, y, z = np.transpose(data)
        return x, y, z

    def make_figure(self, title: Optional[str] = None, **kwargs) -> go.Figure:
        fig = go.Figure(**kwargs)
        camera = dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=-1.5, y=0.7, z=1.4)
        )
        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=dict(
                aspectmode='data',
                camera=camera,
                dragmode='orbit'
            ),
            scene_camera=camera,
            title=title
        )
        return fig

    def plot(self, *args: ChunkGrid, **kwargs):
        fig = self.make_figure()
        for grid in args:
            fig.add_trace(self.make_mesh(grid, **kwargs))
        return fig


if __name__ == '__main__':
    from render.cloud_render import CloudRender
    from model.model_pts import FixedPtsModels
    from filters.fill import flood_fill_at

    # data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    data = FixedPtsModels.bunny()

    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    data_delta_max = np.max(data_max - data_min)

    resolution = 64

    grid = ChunkGrid(16, dtype=int, fill_value=0)
    scaled = (data - data_min) * resolution / data_delta_max
    assert scaled.shape[1] == 3

    grid[scaled] = 1

    # Add padding
    filled = set(tuple(c.index) for c in grid.chunks)
    extra = set(tuple(n) for i in grid.chunks.keys() for f, n in grid.iter_neighbors_indices(i))
    for e in extra:
        grid.ensure_chunk_at_index(e)

    fill_mask = flood_fill_at((7, 9, 7), grid == 0)
    grid[fill_mask] = 3

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(grid == 1, opacity=1.0, flatshading=True))
    fig.add_trace(ren.grid_wireframe(grid == 1, opacity=1.0, size=2.0))
    # fig.add_trace(ren.grid_voxel(grid == 3, opacity=0.1, flatshading=True))
    fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=1)))
    fig.show()
