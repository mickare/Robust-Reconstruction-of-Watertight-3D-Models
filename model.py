import os


class Model:

    @classmethod
    def load_ply_conf(cls, path: str):
        assert os.path.isfile(path)

        with open(path, mode='tr') as fp:
            conf = fp.readlines()

        for line in conf:
            if line.startswith("camera"):
                pass
            elif line.startswith("mesh"):
                pass
            elif line.startswith("bmesh"):
                l = line.split(" ")
                file = l[1]


import plyfile
import plotly.graph_objects as go

data = plyfile.PlyData.read('models/dragon_stand/dragonStandRight_192.ply')

vertex = data["vertex"]
x = vertex.data["x"]
y = vertex.data["y"]
z = vertex.data["z"]

camera = dict(
    up=dict(x=0, y=1, z=0)
)
fig = go.Figure(
    data=[
        go.Scatter3d(x=x, y=z, z=y,
                     mode='markers', marker=dict(color='red', size=1, ))
    ]
)
fig.update_layout(
    yaxis=dict(scaleanchor="x", scaleratio=1),
    scene=dict(
        aspectmode='data',
        xaxis_title='X',
        yaxis_title='Y',  # Flip Y/Z
        zaxis_title='Z',  # Flip Y/Z
        camera=camera,
        dragmode='turntable'
    ),
    scene_camera=camera
)
fig.show()
