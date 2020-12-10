from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import tqdm

from model_mesh import MeshModelLoader


def render_cloud(*args: np.ndarray):
    assert args
    def to_scatter(pts: np.ndarray, **kwargs):
        # Flip Y/Z
        x, z, y = pts.T
        return go.Scatter3d(x=x, y=y, z=z, **kwargs)

    fig = go.Figure(data=[to_scatter(d, mode='markers', marker=dict(size=0.5)) for d in args])
    camera = dict(
        up=dict(x=0, y=1, z=0)
    )
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Z',  # Flip Y/Z
            zaxis_title='Y',  # Flip Y/Z
            camera=camera,
            dragmode='turntable'
        ),
        scene_camera=camera
    )
    fig.show()


if __name__ == '__main__':
    data = MeshModelLoader().load("models/Donut/Donut.obj")
    render_cloud(data)
