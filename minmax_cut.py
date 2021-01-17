import dataclasses
import enum
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Set, Tuple, Optional, TypeVar

from data.chunks import ChunkGrid
from data.faces import ChunkFace
from make_crust import scale_model, get_crust, get_diffusion
from mathlib import Vec3i
from model.model_pts import FixedPtsModels
from render.voxel_render import VoxelRender

TVec3i = Tuple[int, int, int]
V = TypeVar('V')


class CrustFace(enum.IntEnum):
    OUTER = 0
    INNER = 1


@dataclass
class EntryValue:
    __slots__ = ("pos", "origin", "face", "distance")
    pos: Vec3i
    origin: CrustFace
    face: Optional[ChunkFace]
    distance: float


@dataclass(order=True)
class Entry:
    priority: float
    value: EntryValue = dataclasses.field(compare=False)


def add(s: Set[V], value: V) -> bool:
    n = len(s)
    s.add(value)
    return n != len(s)


def minmax_cut(weights: ChunkGrid[float], mask: ChunkGrid[bool], outer: ChunkGrid[bool], inner: ChunkGrid[bool]) \
        -> ChunkGrid[bool]:
    visited_inner: Set[TVec3i] = set()
    visited_outer: Set[TVec3i] = set()
    queue = PriorityQueue()

    result: ChunkGrid[bool] = ChunkGrid(mask.chunk_size, dtype=bool, fill_value=False)

    def enque(visited: Set[TVec3i], entry: EntryValue):
        weight = weights.get_value(entry.pos)
        for f in ChunkFace:
            if f != entry.face:
                next_pos = entry.pos + f.direction()
                if tuple(next_pos) not in visited and mask.get_value(next_pos):
                    if f.flip() == entry.face:
                        w = 2 * weight
                    else:
                        w = weight
                    # w = weight
                    distance = entry.distance + w
                    queue.put_nowait(Entry(distance, EntryValue(pos, entry.origin, f, distance)))

    for pos, w in weights.items(mask=inner):
        enque(visited_inner, EntryValue(pos, CrustFace.INNER, None, 0))
    for pos, w in weights.items(mask=outer):
        enque(visited_outer, EntryValue(pos, CrustFace.OUTER, None, 0))

    count = 0
    while not queue.empty():
        item: Entry = queue.get_nowait()
        tpos = tuple(item.value.pos)
        if item.value.origin == CrustFace.INNER:
            if tpos in visited_outer:
                result.set_value(tpos, True)
            elif add(visited_inner, tpos):
                enque(visited_inner, item.value)
        else:
            if tpos in visited_inner:
                result.set_value(tpos, True)
            elif add(visited_outer, tpos):
                enque(visited_outer, item.value)
        count += 1
    print(count)

    return result


if __name__ == '__main__':
    # data = PtsModelLoader().load("models/bunny/bunnyData.pts")
    # data = PlyModelLoader().load("models/dragon_stand/dragonStandRight.conf")
    # data = MeshModelLoader(samples=30000, noise=0.1).load("models/cat/cat_reference.obj")
    data = FixedPtsModels.bunny()

    num_revert_steps, max_color = 5, 3  # bunny
    # num_revert_steps, max_color = 5, 3  # dragon
    # num_revert_steps, max_color = 5, 3  # cat

    verbose = 2
    chunk_size = 16
    max_steps = 10

    model_pts, model_offset, model_scale = scale_model(data, resolution=64)
    model = ChunkGrid(chunk_size, dtype=bool, fill_value=False)
    model[model_pts] = True
    # Add a chunk layer around the model (fast-flood-fill can wrap the model immediately)
    model.pad_chunks(1)

    crust, outer, inner = get_crust(model, max_steps)
    diffusion = get_diffusion(crust, model)

    cut = minmax_cut(diffusion, crust, outer, inner)

    ren = VoxelRender()
    fig = ren.make_figure()
    # fig.add_trace(ren.grid_voxel(crust, opacity=0.1, name='Crust'))

    scat_kwargs = dict(
        marker=dict(
            size=1.0,
            colorscale=[[0.0, 'rgb(0,0,0)'], [1.0, 'rgb(255,255,255)']],
            cmin=0.0,
            cmax=1.0
        ),
        mode="markers"
    )

    # fig.add_trace(CloudRender().make_value_scatter(diffusion, mask=crust,
    #                                                name="Diffusion", **scat_kwargs))

    fig.add_trace(VoxelRender().grid_voxel(cut, opacity=1.0))

    fig.add_trace(VoxelRender().grid_voxel(inner, opacity=0.1))
    fig.add_trace(VoxelRender().grid_voxel(outer, opacity=0.1))

    fig.show()
