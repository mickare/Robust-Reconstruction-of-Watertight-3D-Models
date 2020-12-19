from typing import Tuple, Sequence, Iterable, Optional, List, Dict

import numpy as np
import plotly.graph_objects as go
import tqdm

from data.chunks import Chunk, ChunkGrid, ChunkType, ChunkFace
from model.model_mesh import MeshModelLoader
from operators.fill_operator import FillOperator


class MeshHelper:
    @classmethod
    def reduce_mesh(cls, vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not vertices or not faces:
            return np.empty((0, 3), dtype=np.float), np.empty((0, 3), dtype=np.int)
        vs = np.vstack(vertices)
        fs = np.vstack(list(cls.join_faces(faces, vertices)))
        vs2, inv = np.unique(vs, return_inverse=True, axis=0)
        fs2 = inv[fs]
        return vs2, fs2

    @classmethod
    def join_faces(cls, faces: Sequence[np.ndarray], vertices: Sequence[np.ndarray]) -> Iterable[np.ndarray]:
        index = 0
        for f, v in zip(faces, vertices):
            yield f + index
            index += len(v)

    @classmethod
    def chunk_to_voxel_mesh(cls, chunk: Chunk, parent: Optional[ChunkGrid] = None,
                            **mask_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        print(mask_kwargs)
        if chunk.empty():
            return np.empty((0, 3), dtype=np.float), np.empty((0, 3), dtype=np.int)
        elif chunk.type == ChunkType.FILL:
            if chunk.is_filled(**mask_kwargs):
                vertices = np.array([
                    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
                ], dtype=np.float) * chunk.size
                faces = np.array([
                    (0, 1, 2), (3, 2, 1),  # Face
                    (2, 3, 6), (7, 6, 3),  #
                    (3, 1, 7), (5, 7, 1),  #
                    (7, 5, 6), (4, 6, 5),  #
                    (5, 1, 4), (0, 4, 1),  #
                    (4, 0, 6), (2, 6, 0)
                ], dtype=np.int)
                return vertices + chunk.index * chunk.size, faces
        elif chunk.type == ChunkType.ARRAY:
            neighbors: List[Optional[Chunk]] = [None] * 6
            if parent:
                neighbors = [c for f, c in parent.iter_neighbors(chunk.index, flatten=False)]
                assert len(neighbors) == 6

            crust = chunk.mask(**mask_kwargs)
            if not crust.any():
                return np.empty((0, 3), dtype=np.float), np.empty((0, 3), dtype=np.int)

            cx = np.array(np.pad(crust, ((1, 1), (0, 0), (0, 0)), constant_values=False), dtype=np.bool)
            cy = np.array(np.pad(crust, ((0, 0), (1, 1), (0, 0)), constant_values=False), dtype=np.bool)
            cz = np.array(np.pad(crust, ((0, 0), (0, 0), (1, 1)), constant_values=False), dtype=np.bool)

            def transfer_face(face: ChunkFace, dst: np.ndarray):
                n: Optional[Chunk] = neighbors[face]
                if n is not None:
                    dst[face.slice()] = n.mask(**mask_kwargs)[face.flip().slice()]

            # X-Neighbors
            transfer_face(ChunkFace.NORTH, cx)
            transfer_face(ChunkFace.SOUTH, cx)

            # Y-Neighbors
            transfer_face(ChunkFace.EAST, cy)
            transfer_face(ChunkFace.WEST, cy)

            # Z-Neighbors
            transfer_face(ChunkFace.TOP, cz)
            transfer_face(ChunkFace.BOTTOM, cz)

            dx = cx[1:, :, :] ^ cx[:-1, :, :]
            dy = cy[:, 1:, :] ^ cy[:, :-1, :]
            dz = cz[:, :, 1:] ^ cz[:, :, :-1]

            ix = np.nonzero(dx)
            iy = np.nonzero(dy)
            iz = np.nonzero(dz)

            faces_front = np.array([(0, 1, 2), (3, 2, 1)], dtype=np.int)
            faces_back = np.array([(3, 2, 1), (0, 1, 2)], dtype=np.int)

            vx = [np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]) + f for f, d in zip(np.transpose(ix), dx[ix])]
            fx = [(faces_front if d >= 0 else faces_back) + (n * 4) for n, d in enumerate(dx[ix])]

            vy = [np.array([(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]) + f for f, d in zip(np.transpose(iy), dy[iy])]
            fy = [(faces_front if d >= 0 else faces_back) + (n * 4) for n, d in enumerate(dy[iy])]

            vz = [np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]) + f for f, d in zip(np.transpose(iz), dz[iz])]
            fz = [(faces_front if d >= 0 else faces_back) + (n * 4) for n, d in enumerate(dz[iz])]

            vertices, faces = cls.reduce_mesh([np.vstack(v) for v in (vx, vy, vz) if v],
                                              [np.vstack(f) for f in (fx, fy, fz) if f])
            return vertices + chunk.index * chunk.size, faces
            # end elif chunk.type == ChunkType.ARRAY:

        # else empty
        return np.empty((0, 3), dtype=np.float), np.empty((0, 3), dtype=np.int)

    @classmethod
    def grid_to_voxel_mesh(cls, grid: ChunkGrid, verbose=True, **mask_kwargs):
        if verbose:
            chunks = tqdm.tqdm(grid.chunks, desc="Building voxel mesh")
        else:
            chunks = grid.chunks
        vertices, faces = zip(*(cls.chunk_to_voxel_mesh(c, **mask_kwargs) for c in chunks))
        return cls.reduce_mesh(vertices, faces)


class VoxelRender:

    def __init__(self, flip_zy=True):
        self.flip_zy = flip_zy

    def _unwrap(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = pts.T
        if self.flip_zy:
            return x, z, y
        return x, y, z

    def make_mesh(self, grid: ChunkGrid, verbose=True, voxel_kwargs: Optional[Dict] = None, **kwargs):
        kwargs.setdefault("flatshading", True)
        kwargs.setdefault("lighting", dict(
            ambient=0.4,
            # diffuse=0.5,
            # facenormalsepsilon=0.0000000000001,
            # fresnel=0.001,
            # roughness=0.9,
            # specular=0.1,
        ))

        voxel_kwargs = voxel_kwargs or {}
        voxel_kwargs.setdefault("verbose", True)
        vertices, faces = MeshHelper.grid_to_voxel_mesh(grid, **voxel_kwargs)
        x, y, z = self._unwrap(vertices)
        i, j, k = faces.T
        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)

    def make_figure(self, **kwargs) -> go.Figure:
        fig = go.Figure()
        camera = dict(
            up=dict(x=0, y=1, z=0)
        )
        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y' if not self.flip_zy else 'Z',
                zaxis_title='Z' if not self.flip_zy else 'Y',
                camera=camera,
                dragmode='turntable'
            ),
            scene_camera=camera
        )
        fig.update_traces(lightposition=dict(X=1000, Y=1000, Z=1000))
        return fig

    def plot(self, *args: ChunkGrid, **kwargs):
        fig = self.make_figure()
        for grid in args:
            fig.add_trace(self.make_mesh(grid, **kwargs))
        return fig


def plot_boolean_chunkgrid():
    from render_cloud import CloudRender

    # data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    # data = PlyModelLoader().load("models/dragon_stand/dragonStandRight.conf")
    data = MeshModelLoader(samples=30000, noise=0.1).load("models/cat/cat_reference.obj")

    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    data_delta_max = np.max(data_max - data_min)

    resolution = 32

    grid = ChunkGrid(8, empty_value=False)
    scaled = (data - data_min) * resolution / data_delta_max
    assert scaled.shape[1] == 3

    for p in scaled:
        pos = np.array(p, dtype=int)
        c = grid.ensure_chunk_at_pos(pos)
        c.set_pos(pos, True)

    c = grid.ensure_chunk_at_pos((-2, -2, 0))
    c.set_fill(True)

    fig = VoxelRender().plot(grid, opacity=1.0, flatshading=True, voxel_kwargs=dict(logic=True))
    fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
    fig.show()


if __name__ == '__main__':
    from render_cloud import CloudRender

    # data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    # data = PlyModelLoader().load("models/dragon_stand/dragonStandRight.conf")
    data = MeshModelLoader(samples=30000, noise=0.1).load("models/cat/cat_reference.obj")

    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    data_delta_max = np.max(data_max - data_min)

    resolution = 32

    grid = ChunkGrid(8, empty_value=0)
    scaled = (data - data_min) * resolution / data_delta_max
    assert scaled.shape[1] == 3

    for p in scaled:
        pos = np.array(p, dtype=int)
        c = grid.ensure_chunk_at_pos(pos)
        c.set_pos(pos, 1)

    # Add padding
    filled = set(tuple(c.index) for c in grid.chunks)
    extra = set(tuple(n) for i in grid.chunks.keys() for f, n in grid.iter_neighbors_indicies(i))
    for e in extra:
        grid.ensure_chunk_at_index(e)

    fill = FillOperator(grid)
    # c = grid.ensure_chunk_at_pos((7, 9, 7))
    # c.set_fill(2)
    masks = fill.fill_masks_parallel((7, 9, 7))

    for m in masks.values():
        m.apply(grid.ensure_chunk_at_index(m.index), 3)

    ren = VoxelRender()
    fig = ren.make_figure()
    # fig.add_trace(ren.make_mesh(grid, opacity=0.4, flatshading=True, voxel_kwargs=dict(value=1)))
    fig.add_trace(ren.make_mesh(grid, opacity=0.4, flatshading=True, voxel_kwargs=dict(value=2)))
    fig.add_trace(ren.make_mesh(grid, opacity=0.1, flatshading=True, voxel_kwargs=dict(value=3)))
    fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
    fig.show()
