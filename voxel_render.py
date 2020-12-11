from typing import Tuple, Sequence, Iterable, Optional

import numpy as np
import plotly.graph_objects as go

from tree import ChunkGrid, Chunk, ChunkSize


class MeshHelper:
    @classmethod
    def reduce_mesh(cls, vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        vs = np.vstack(vertices)
        fs = np.vstack(cls.join_faces(faces, vertices))
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
    def chunk_to_voxel_mesh(cls, chunk: Chunk, parent: Optional[ChunkGrid] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Check for neighbors
        neighbors = [None] * 6
        if parent:
            neighbors = list(parent.iter_neighbors(chunk.index, flatten=False))
            assert len(neighbors) == 6

        if isinstance(chunk.crust, np.ndarray):

            cx = np.array(np.pad(chunk.crust, ((1, 1), (0, 0), (0, 0)), constant_values=False), dtype=np.bool)
            cy = np.array(np.pad(chunk.crust, ((0, 0), (1, 1), (0, 0)), constant_values=False), dtype=np.bool)
            cz = np.array(np.pad(chunk.crust, ((0, 0), (0, 0), (1, 1)), constant_values=False), dtype=np.bool)

            # X-Neighbors
            if neighbors[0]:
                cx[0, :, :] = neighbors[0].crust_voxels()[-1, :, :]
            if neighbors[1]:
                cx[-1, :, :] = neighbors[1].crust_voxels()[0, :, :]

            # Y-Neighbors
            if neighbors[2]:
                cy[:, 0, :] = neighbors[2].crust_voxels()[:, -1, :]
            if neighbors[3]:
                cy[:, -1, :] = neighbors[3].crust_voxels()[:, 0, :]

            # Z-Neighbors
            if neighbors[4]:
                cz[:, :, 0] = neighbors[4].crust_voxels()[:, :, -1]
            if neighbors[5]:
                cz[:, :, -1] = neighbors[5].crust_voxels()[:, :, 0]

            dx = cx[1:, :, :] ^ cx[:-1, :, :]
            dy = cy[:, 1:, :] ^ cy[:, :-1, :]
            dz = cz[:, :, 1:] ^ cz[:, :, :-1]

            ix = np.nonzero(dx)
            iy = np.nonzero(dy)
            iz = np.nonzero(dz)

            vx = [np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]) + f for f, d in zip(np.transpose(ix), dx[ix])]
            fx = [(np.array([(0, 1, 2), (3, 2, 1)]) if d >= 0 else np.array([(3, 2, 1), (0, 1, 2)])) + (n * 4)
                  for n, d in enumerate(dx[ix])]

            vy = [np.array([(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]) + f for f, d in zip(np.transpose(iy), dy[iy])]
            fy = [(np.array([(0, 1, 2), (3, 2, 1)]) if d >= 0 else np.array([(3, 2, 1), (0, 1, 2)])) + (n * 4)
                  for n, d in enumerate(dy[iy])]

            vz = [np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]) + f for f, d in zip(np.transpose(iz), dz[iz])]
            fz = [(np.array([(0, 1, 2), (3, 2, 1)]) if d >= 0 else np.array([(3, 2, 1), (0, 1, 2)])) + (n * 4)
                  for n, d in enumerate(dz[iz])]

            # vvx = np.vstack(vx)
            # vvy = np.vstack(vy)
            # vvz = np.vstack(vz)

            # vertices = np.vstack((vvx, vvy, vvz)) + chunk.index * ChunkSize
            # faces = np.vstack((
            #     np.vstack(fx),
            #     np.vstack(fy) + len(vvx),
            #     np.vstack(fz) + len(vvx) + len(vvy)
            # ))
            # vertices, inv = np.unique(vertices, return_inverse=True, axis=0)
            # faces = inv[faces]
            # return vertices, faces
            vertices, faces = cls.reduce_mesh([np.vstack(vx), np.vstack(vy), np.vstack(vz)],
                                              [np.vstack(fx), np.vstack(fy), np.vstack(fz)])

            return vertices + chunk.index * ChunkSize, faces

        elif bool(chunk.crust):
            vertices = np.array([
                (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
            ], dtype=np.float) * ChunkSize
            faces = np.array([
                (0, 1, 2), (3, 2, 1),  # Face
                (2, 3, 6), (7, 6, 3),  #
                (3, 1, 7), (5, 7, 1),  #
                (7, 5, 6), (4, 6, 5),  #
                (5, 1, 4), (0, 4, 1),  #
                (4, 0, 6), (2, 6, 0)
            ], dtype=np.int)
            return vertices, faces
        else:
            raise RuntimeError("invalid chunk distance")

    @classmethod
    def grid_to_voxel_mesh(cls, grid: ChunkGrid):
        vertices, faces = zip(*(cls.chunk_to_voxel_mesh(c, parent=grid) for c in grid.chunks.values()))
        return cls.reduce_mesh(vertices, faces)


class VoxelRender:

    def __init__(self, flip_zy=True):
        self.flip_zy = flip_zy

    def _unwrap(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = pts.T
        if self.flip_zy:
            return x, z, y
        return x, y, z

    def make_mesh(self, grid: ChunkGrid, **kwargs):
        kwargs.setdefault("flatshading", True)

        vertices, faces = MeshHelper.grid_to_voxel_mesh(grid)
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
                yaxis_title='Y',
                zaxis_title='Z',
                camera=camera,
                dragmode='turntable'
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
    from cloud_render import CloudRender
    from model_pts import PtsModelLoader

    data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)

    grid = ChunkGrid(16)
    scaled = (data - data_min) / np.max(data_max - data_min) * grid.resolution
    assert scaled.shape[1] == 3

    for p in scaled:
        pos = np.array(p, dtype=int)
        c = grid.create_if_absent(pos)
        c.set_crust(pos, True)

    fig = VoxelRender().plot(grid, opacity=0.4)
    fig.add_trace(CloudRender().make_scatter(scaled, marker=dict(size=0.5)))
    fig.show()
