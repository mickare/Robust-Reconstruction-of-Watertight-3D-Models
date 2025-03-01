"""
Main example that
"""

from typing import Optional, Sequence

import numba
import numpy as np
import pytorch3d.structures
import torch

from example import Example, example_config, example_load
from reconstruction import mesh_extraction
from reconstruction.data.chunks import ChunkGrid
from reconstruction.filters.dilate import dilate
from reconstruction.mathlib import Vec3i
from reconstruction.medial_axis_propagating import crust_fix
from reconstruction.mincut import MinCut
from reconstruction.reconstruction import scale_model, crust_dilation, plot_voxels, diffuse, fill_components, \
    cleanup_components
from reconstruction.render.cloud_render import CloudRender
from reconstruction.render.voxel_render import VoxelRender
from reconstruction.utils import timed

numba.config.THREADING_LAYER = 'omp'

# Configuration, modify here to change the model
CHUNKSIZE = 16
RESOLUTION_INIT = 64
example = Example.BunnyFixed
STEPS = 4
APPROX_MEDIAL_AXIS = True

if __name__ == '__main__':
    # Set initial resolution
    resolution = RESOLUTION_INIT

    print("Loading model")
    with timed("\tTime: "):

        data = example_load(example)
        cfg = example_config[example]
        dilations_max = cfg["dilations_max"]
        dilations_reverse = cfg["dilations_reverse"]

        data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
        model: ChunkGrid[np.bool8] = ChunkGrid(CHUNKSIZE, dtype=np.bool8, fill_value=np.bool8(False))
        model[data_pts] = True
        model.pad_chunks(2)
        model.cleanup()

    plot_model: Optional[np.ndarray] = data_pts[::5]

    crust: ChunkGrid[np.bool8] = model.copy()
    crust.cleanup(remove=True)

    # ren = VoxelRender()
    # fig = ren.make_figure()
    # fig.add_trace(ren.grid_voxel(initial_crust, opacity=0.1, name='Initial'))
    # fig.add_trace(CloudRender().make_scatter(data_pts, size=1, name='Model'))
    # fig.show()

    print("Dilation")
    with timed("\tTime: "):
        crust, components, dilation_step = crust_dilation(crust, max_steps=dilations_max,
                                                          reverse_steps=dilations_reverse)
        # assert components._fill_value == 2

        plot_voxels(components == 0, components, title=f"Initial Dilation").show()
        crust_dilate = dilate(crust)
        outer_fill = components == 2
        crust_outer = outer_fill & crust_dilate
        crust_inner = (components == 3) & crust_dilate

        assert crust_dilate._fill_value == False
        assert outer_fill._fill_value == True
        assert crust_outer._fill_value == False
        assert crust_inner._fill_value == False

    """
    Increase resolution and make the crust_fixmesh approximation finer
    """
    for resolution_step in range(0, STEPS):
        print(f"RESOLUTION STEP: {resolution_step}")

        if APPROX_MEDIAL_AXIS:
            """
            Approximate Voxel near Medial Axis, by propagating a Normal field inwards.
            Then for each voxel compute a normal cone and mark the voxel as inner component when the cone angle is greater than 90°.
            """
            print("Crust-Fix")
            with timed("\tTime: "):
                crust_inner |= crust_fix(
                    crust, outer_fill, crust_outer, crust_inner,
                    min_distance=dilation_step,
                    data_pts=plot_model
                )
            #     # crust_inner[model] = False  # Remove model voxels if they have been added by the crust fix

        print("Render Crust")
        with timed("\tTime: "):
            ren = VoxelRender()
            fig = ren.make_figure(title=f"Step-{resolution_step}: Crust")
            fig.add_trace(ren.grid_voxel(crust_outer, opacity=0.1, name='Outer'))
            fig.add_trace(ren.grid_voxel(crust_inner, opacity=1.0, name='Inner'))
            if plot_model is not None:
                fig.add_trace(CloudRender().make_scatter(plot_model, size=0.7, name='Model'))
            fig.show()

        print("Diffusion")
        with timed("\tTime: "):
            diff = diffuse(model, repeat=3)

        print("Plot-Diffusion")
        with timed("\tTime: "):
            ren = CloudRender()
            fig = ren.make_figure()

            # Cut in half
            diff_mask = (diff != 1.0) & crust
            half = (np.max(data_pts, axis=0) + np.min(data_pts, axis=0)).astype(int) // 2
            half_chunk = half // diff_mask.chunk_size
            half_chunk_split = half[2] % diff_mask.chunk_size
            for index in list(diff_mask.chunks.keys()):
                if index[2] > half_chunk[2]:
                    del diff_mask.chunks[index]
                elif index[2] == half_chunk[2]:
                    ch = diff_mask.chunks.get(index)
                    arr = ch.to_array()
                    arr[:, :, half_chunk_split:] = False
                    ch.set_array(arr)

            items = list(diff.items(mask=diff_mask))
            items.sort(key=lambda e: e[0][2] * 1024 + e[0][1] + e[0][0])
            points, values = zip(*items)  # type: Sequence[Vec3i], Sequence
            pts = np.array(points, dtype=np.float32) + 0.5

            fig.add_trace(ren.make_scatter(
                pts,
                name="Diffusion",
                marker=dict(
                    size=2.0,
                    opacity=0.7,
                    colorscale='Viridis',
                    color=np.array(values)
                ),
                mode="markers",
            ))
            fig.show()

        print("MinCut")
        with timed("\tTime: "):
            mincut = MinCut(diff, crust, crust_outer, crust_inner)
            segment0, segment1 = mincut.grid_segments()
            thincrust = segment0 & segment1

        print("Render")
        with timed("\tTime: "):
            ren = VoxelRender()
            fig = ren.make_figure(title=f"Step-{resolution_step}: Segments")
            fig.add_trace(ren.grid_voxel(segment0, opacity=0.1, name='Segment 0'))
            fig.add_trace(ren.grid_voxel(segment1, opacity=0.1, name='Segment 1'))
            fig.add_trace(ren.grid_voxel(thincrust, opacity=1.0, name='Join'))
            if plot_model is not None:
                fig.add_trace(CloudRender().make_scatter(plot_model, size=1, name='Model'))
            fig.show()

        print("Volumetric refinement")
        with timed("\tTime: "):
            # Rebuild model
            resolution *= 2
            data_pts, data_offset, data_scale = scale_model(data, resolution=resolution)
            model = ChunkGrid(CHUNKSIZE, np.bool8, fill_value=np.bool8(False))
            model[data_pts] = np.bool8(True)

            plot_model: Optional[np.ndarray] = data_pts[::5]

            # Build new crust
            crust = dilate(dilate(thincrust.split(2), steps=1) | dilate(model, steps=3))
            crust.cleanup(remove=True)

            components, count = fill_components(crust, max_components=5)
            cleanup_components(crust, components, count)

            outer_fill = (components == 2)
            outer_fill.cleanup(remove=True)

            crust_dilate = dilate(crust)
            crust_outer = outer_fill & crust_dilate
            crust_inner = (components == 3) & crust_dilate

            crust_outer.cleanup(remove=True)
            crust_inner.cleanup(remove=True)

            dilation_step = 2

            # Validate data
            assert crust._fill_value == False
            assert outer_fill._fill_value == True
            assert crust_outer._fill_value == False
            assert crust_inner._fill_value == False

        print("Extract mesh")
        with timed("\tTime: "):
            # Extraction
            mesh_extractor = mesh_extraction.MeshExtraction(mincut)
            vertices, faces = mesh_extractor.extract_mesh()

            ren = VoxelRender()
            fig = ren.make_figure()
            fig.add_trace(ren.make_mesh(vertices, faces, name='Mesh', flatshading=True))
            fig.add_trace(ren.make_wireframe(vertices, faces, name='Wireframe'))
            fig.update_layout(showlegend=True)
            fig.show()

        print("Smoothing mesh")
        with timed("\tTime: "):
            # Smoothing
            pytorch_mesh = pytorch3d.structures.Meshes(verts=[torch.FloatTensor(vertices)],
                                                       faces=[torch.LongTensor(faces)])

            smoothed_vertices = mesh_extraction.Smoothing().smooth(vertices, faces, diff, pytorch_mesh)
            verts = smoothed_vertices.cpu().detach().numpy()
            faces = torch.cat(pytorch_mesh.faces_list()).cpu().detach().numpy()

            ren = VoxelRender()
            fig = ren.make_figure()
            fig.add_trace(ren.make_mesh(verts, faces, name='Mesh', flatshading=False))
            fig.add_trace(ren.make_wireframe(verts, faces, name='Wireframe'))
            fig.update_layout(showlegend=True)
            fig.show()
