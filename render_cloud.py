from typing import Tuple, Optional, List

import numpy as np
import plotly.graph_objects as go

from data.chunks import ChunkGrid, Chunk
from mathlib import Vec3i
from model.model_mesh import MeshModelLoader
from utils import merge_default


class CloudRender:

    @classmethod
    def _unwrap(cls, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(data) == 0:
            return np.empty(0), np.empty(0), np.empty(0)
        x, y, z = np.transpose(data)
        return x, y, z

    def make_scatter(self, pts: np.ndarray, size=0.5, **kwargs):
        merge_default(kwargs, mode='markers', marker=dict(size=size))
        x, y, z = self._unwrap(pts)
        return go.Scatter3d(x=x, y=y, z=z, **kwargs)

    def make_figure(self, **kwargs) -> go.Figure:
        fig = go.Figure()
        camera = dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=-1, y=-1, z=0.5)
        )
        yaxis = dict()
        zaxis = dict()
        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=dict(
                aspectmode='data',
                yaxis=yaxis,
                zaxis=zaxis,
                xaxis_title='X',
                camera=camera,
                dragmode='turntable'
            ),
            scene_camera=camera
        )
        return fig

    def make_value_scatter(self, grid: ChunkGrid, mask: ChunkGrid[bool], **kwargs):

        items = list(grid.items(mask=mask))
        points, values = zip(*items)  # type: Sequence[Vec3i], Sequence
        pts = np.array(points, dtype=np.float32) + 0.5
        values = np.array(values)

        merge_default(kwargs, marker=dict(color=values))
        return self.make_scatter(pts, **kwargs)

    def plot(self, *args: np.ndarray, size=0.5, **kwargs):
        fig = self.make_figure()
        merge_default(kwargs, mode='markers', marker=dict(size=size))
        for d in args:
            fig.add_trace(self.make_scatter(d, **kwargs))
        return fig


if __name__ == '__main__':
    data = MeshModelLoader(20000, noise=0.1).load("models/cat/cat_reference.obj")
    fig = CloudRender().plot(data)
    fig.show()
