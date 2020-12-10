import plotly.graph_objects as go
import tqdm

from model_ply import ScanFolder

scans = ScanFolder.load_ply_conf("models/dragon_stand/dragonStandRight.conf")

camera = dict(
    up=dict(x=0, y=1, z=0)
)
fig = go.Figure(
    data=[
        s.scatter(transf=scans.transform, mode='markers', marker=dict(size=0.5))
        for s in tqdm.tqdm(scans.scans, desc="Load files")
    ]
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
