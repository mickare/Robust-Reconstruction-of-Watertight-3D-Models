from typing import Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from reconstruction.data.chunks import ChunkGrid
from reconstruction.mathlib import Vec3i
from reconstruction.model.model_mesh import MeshModelLoader
from reconstruction.utils import merge_default


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

    def make_figure(self, title=None, **kwargs) -> go.Figure:
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
    data = MeshModelLoader(20000, noise=0.1).load("../models/cat/cat_reference.obj")
    fig = CloudRender().plot(data)
    fig.show()
