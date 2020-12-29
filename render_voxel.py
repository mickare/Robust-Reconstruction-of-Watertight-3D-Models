from typing import Tuple, Sequence, Iterable, Optional, List

import numpy as np
import plotly.graph_objects as go
import tqdm

from data.chunks import Chunk, ChunkGrid, ChunkFace
from mathlib import Vec3f
from utils import merge_default, timed


def immutable(arr: np.ndarray):
    arr.flags.writeable = False
    return arr


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
    def reduce_mesh(cls, vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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
    def _create_mesh_from_deltas(cls, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray):
        ix = np.nonzero(dx)
        iy = np.nonzero(dy)
        iz = np.nonzero(dz)

        # Local variable cache
        faces_front = cls._faces_front
        faces_back = cls._faces_back
        vert_x = cls._vert_x
        vert_y = cls._vert_y
        vert_z = cls._vert_z

        vx = [vert_x + f for f, d in zip(np.transpose(ix), dx[ix])]
        fx = [(faces_front if d >= 0 else faces_back) + (n * 4) for n, d in enumerate(dx[ix])]

        vy = [vert_y + f for f, d in zip(np.transpose(iy), dy[iy])]
        fy = [(faces_front if d >= 0 else faces_back) + (n * 4) for n, d in enumerate(dy[iy])]

        vz = [vert_z + f for f, d in zip(np.transpose(iz), dz[iz])]
        fz = [(faces_front if d >= 0 else faces_back) + (n * 4) for n, d in enumerate(dz[iz])]

        vertices, faces = cls.reduce_mesh([np.vstack(v) for v in (vx, vy, vz) if v],
                                          [np.vstack(f) for f in (fx, fy, fz) if f])
        return vertices, faces

    @classmethod
    def extract_voxel_mesh(cls, mask: np.ndarray, neighbors: Sequence[Optional[Chunk]] = None):
        if neighbors is None:
            neighbors = [None] * 6

        voxels = np.pad(mask.astype(dtype=np.int8), 1, constant_values=0)

        def transfer_face(face: ChunkFace, dst: np.ndarray):
            n: Optional[Chunk] = neighbors[face]
            if n is not None:
                if n.is_filled():
                    dst[face.slice(other=slice(1, -1))] = np.bool8(n.value)
                else:
                    dst[face.slice(other=slice(1, -1))] = n.to_array()[face.flip().slice()].astype(dtype=np.bool8)

        # X-Neighbors
        transfer_face(ChunkFace.NORTH, voxels)
        transfer_face(ChunkFace.SOUTH, voxels)
        # Y-Neighbors
        transfer_face(ChunkFace.TOP, voxels)
        transfer_face(ChunkFace.BOTTOM, voxels)
        # Z-Neighbors
        transfer_face(ChunkFace.EAST, voxels)
        transfer_face(ChunkFace.WEST, voxels)

        # Directions of faces -1 and +1, no face is zero
        dx = voxels[1:, 1:-1, 1:-1] - voxels[:-1, 1:-1, 1:-1]
        dy = voxels[1:-1, 1:, 1:-1] - voxels[1:-1, :-1, 1:-1]
        dz = voxels[1:-1, 1:-1, 1:] - voxels[1:-1, 1:-1, :-1]

        # Remove duplicate faces at border, show only faces that point away
        def clip_face(face0: ChunkFace, dst: np.ndarray):
            s0 = face0.slice()
            s1 = face0.flip().slice()
            dst[s0] = np.where(dst[s0] > 0, 0, dst[s0])
            dst[s1] = np.where(dst[s1] < 0, 0, dst[s1])

        clip_face(ChunkFace.NORTH, dx)
        clip_face(ChunkFace.TOP, dy)
        clip_face(ChunkFace.EAST, dz)

        return cls._create_mesh_from_deltas(dx, dy, dz)

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
            vertices, faces = zip(*(cls.chunk_to_voxel_mesh(c, parent=grid, **kwargs) for c in chunks))
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

    def make_mesh(self, vertices: np.ndarray, faces: np.ndarray,
                  scale=1.0, offset: Optional[Vec3f] = None, **kwargs):
        merge_default(kwargs, **self.default_mesh_kwargs)
        kwargs.setdefault("flatshading", True)

        offset = (0, 0, 0) if offset is None else offset

        vertices = scale * vertices + offset
        x, y, z = self._unwrap(vertices)
        i, j, k = self._unwrap(faces)
        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)

    @classmethod
    def _unwrap(cls, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(data) == 0:
            return np.empty(0), np.empty(0), np.empty(0)
        x, y, z = np.transpose(data)
        return x, y, z

    def make_figure(self, **kwargs) -> go.Figure:
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
            scene_camera=camera
        )
        return fig

    def plot(self, *args: ChunkGrid, **kwargs):
        fig = self.make_figure()
        for grid in args:
            fig.add_trace(self.make_mesh(grid, **kwargs))
        return fig


if __name__ == '__main__':
    from render_cloud import CloudRender
    from model.model_pts import PtsModelLoader, FixedPtsModels
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
    extra = set(tuple(n) for i in grid.chunks.keys() for f, n in grid.iter_neighbors_indicies(i))
    for e in extra:
        grid.ensure_chunk_at_index(e)

    fill_mask = flood_fill_at((7, 9, 7), grid == 0)
    grid[fill_mask] = 3

    ren = VoxelRender()
    fig = ren.make_figure()
    fig.add_trace(ren.grid_voxel(grid == 1, opacity=0.5, flatshading=True))
    # fig.add_trace(ren.grid_voxel(grid == 3, opacity=0.1, flatshading=True))
    fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=1)))
    fig.show()
