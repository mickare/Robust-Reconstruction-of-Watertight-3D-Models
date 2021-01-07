import collections
import functools
import sys
from typing import Optional, List, Tuple, Collection, Set

import numpy as np
from scipy import ndimage

from data.chunks import Index, ChunkFace, Chunk, ChunkGrid
from mathlib import Vec3i


class FloodFillTask:
    __slots__ = ["index", "_face", "_image"]

    def __init__(self, index: Index, face: Optional[ChunkFace] = None, image: Optional[np.ndarray] = None):
        assert len(index) == 3
        self.index = tuple(index)
        self._face = face
        self._image = image
        if face is not None:
            assert isinstance(face, ChunkFace)
        if image is not None:
            assert isinstance(image, np.ndarray)

    @property
    def face(self) -> Optional[ChunkFace]:
        return self._face

    def merge(self, other: "FloodFillTask", shape: Tuple[int, ...]):
        assert self.index == other.index
        if self._face is not None and other._face is not None and self._face == other._face:
            return FloodFillTask(self.index, self._face)
        mask = self.image(shape) | other.image(shape)
        return FloodFillTask(self.index, image=mask)

    def image(self, shape: Tuple[int, ...]) -> np.ndarray:
        if self._face is not None:
            m = np.zeros(shape, dtype=bool)
            m[self._face.slice()] = True
            return m
        elif self._image is not None:
            return self._image
        else:
            return np.zeros(shape, dtype=bool)

    def __eq__(self, other):
        if isinstance(other, FloodFillTask):
            if self.index == other.index:
                return self._face == other._face and np.all(self._image == other._image)
        return False

    def __hash__(self):
        return hash(self.index)

    @classmethod
    def merge_all(cls, tasks: Collection["FloodFillTask"], shape: Tuple[int, ...]) -> "FloodFillTask":
        def _merge(a: "FloodFillTask", b: "FloodFillTask"):
            return a.merge(b, shape)

        return functools.reduce(_merge, tasks)


class FloodFillOperator:

    def __init__(self, mask: ChunkGrid, verbose=False):
        self.mask: ChunkGrid[np.bool8] = mask.astype(np.bool8).cleanup()
        self.verbose = verbose
        self.__mask_chunks_get = self.mask.chunks.get

    def _compute_fill_task(self, task: FloodFillTask) -> Optional[Tuple[Chunk[np.bool8], List[FloodFillTask]]]:
        """
        Create a fill mask for a chunk with a source mask
        :param task: fill task
        :return: A tuple of the resulting fill mask for this chunk and tasks for the neighboring chunks to fill
        """
        chunk: Chunk[np.bool8] = self.__mask_chunks_get(task.index)
        if chunk is None:
            return None

        # Method cache (prevent lookup in loop)
        __face_flip = ChunkFace.flip
        __face_slice = ChunkFace.slice

        if chunk.is_filled():
            if chunk.value:
                return (
                    Chunk(task.index, chunk.size, dtype=np.bool8).set_fill(True),
                    [
                        FloodFillTask(i, face=__face_flip(f))
                        for f, i in ChunkGrid.iter_neighbors_indicies(chunk.index)
                        if f != task.face
                    ]
                )
            else:
                # return Chunk(task.index, chunk.size, dtype=np.bool8).set_fill(False), []
                return None

        task_image: np.ndarray = task.image(chunk.array_shape)
        assert task_image.shape == chunk.array_shape
        if not task_image.any():
            # return Chunk(task.index, chunk.size, dtype=np.bool8).set_fill(False), []
            return None
        else:
            mask = chunk.to_array()
            # Enforce that only "free" fields in dst_mask are filled
            img = task_image & mask
            result = ndimage.binary_propagation(img, mask=mask).astype(np.bool8)

            # Create tasks where propagation is possible
            tasks = []
            for f, i in ChunkGrid.iter_neighbors_indicies(chunk.index):
                face_slice = result[__face_slice(f)]
                if face_slice.all():
                    if f != task.face:
                        tasks.append(FloodFillTask(i, face=__face_flip(f)))
                elif face_slice.any():
                    tmp = np.full(chunk.shape, False, dtype=np.bool8)
                    tmp[__face_slice(__face_flip(f))] = face_slice
                    tasks.append(FloodFillTask(i, image=tmp))
            return Chunk(task.index, chunk.size, dtype=np.bool8).set_array(result).cleanup(), tasks

    def _fill_fast(self, image: ChunkGrid[np.bool8], tasks: List[FloodFillTask]):
        queue = collections.deque(tasks)
        rest: List[FloodFillTask] = []
        visited: Set[Index] = set()
        visited_full: Set[Index] = set()

        # Method cache (prevent lookup in loop)
        __visited_add = visited.add
        __visited_full_add = visited_full.add
        __mask_chunks_get = self.__mask_chunks_get
        __image_ensure_chunk_at_index = image.ensure_chunk_at_index
        __iter_neighbors_indicies = ChunkGrid.iter_neighbors_indicies
        __face_flip = ChunkFace.flip
        __queue_popleft = queue.popleft
        __queue_extend = queue.extend
        __rest_append = rest.append
        __chunk_is_filled = Chunk.is_filled
        __chunk_set_fill = Chunk.set_fill

        def add_visited(value) -> bool:
            n = len(visited)
            __visited_add(value)
            return n != len(visited)

        while queue:
            t = __queue_popleft()
            if add_visited(t.index):
                m: Chunk = __mask_chunks_get(t.index, None)
                if m is not None:
                    if __chunk_is_filled(m):
                        if m.value:
                            __visited_full_add(t.index)
                            o = __image_ensure_chunk_at_index(t.index)
                            __chunk_set_fill(o, True)
                            __queue_extend([
                                FloodFillTask(i, face=__face_flip(f)) for f, i in
                                __iter_neighbors_indicies(t.index)
                                if t.face != f
                            ])
                    else:
                        __rest_append(t)
            elif t.index not in visited_full:
                __rest_append(t)
        return rest

    def _iterate_fill(self, image: ChunkGrid[np.bool8], tasks: List[FloodFillTask], max_steps: int, **kwargs) \
            -> ChunkGrid[np.bool8]:

        def merge_chunk_into(src: Chunk[np.bool8], dst: Chunk[np.bool8]) -> bool:
            old = np.copy(dst.value)
            dst |= src
            return np.any(old != dst.value)

        # Method cache (prevent lookup in loop)
        __list_append = list.append
        __task_merge_all = FloodFillTask.merge_all
        __compute_fill_task = self._compute_fill_task
        __image_chunks_get = image.chunks.get

        chunk_shape = self.mask.chunk_shape
        for step in range(1, max_steps + 1):
            if step == max_steps:
                if self.verbose:
                    print("Maximum fill mask steps reached!", sys.stderr)
                break

            # First do fast fill operations (e.g. empty chunk)
            tasks = self._fill_fast(image, tasks)

            # Reduce and Merge tasks
            tasks_indexed = collections.defaultdict(list)
            for t in tasks:
                __list_append(tasks_indexed[tuple(t.index)], t)
            tasks_reduced: List[FloodFillTask] = [__task_merge_all(ts, chunk_shape) for ts in
                                                  tasks_indexed.values()]

            # Rebuild tasks for next iteration
            tasks = []
            for t in tasks_reduced:
                res = __compute_fill_task(t)
                if res is not None:
                    res_mask, res_tasks = res  # type: Chunk[np.bool8], List[FloodFillTask]
                    index = res_mask.index
                    o = __image_chunks_get(index)
                    if o is not None:
                        changed = merge_chunk_into(res_mask, o)
                        o.cleanup()
                        if changed:
                            tasks.extend(t for t in res_tasks if t.index in self.mask.chunks)
            if not tasks:
                break
        return image

    def fill(self, image: ChunkGrid[np.bool8], max_steps: Optional[int] = None, **kwargs) -> ChunkGrid[np.bool8]:
        assert image.chunk_size == self.mask.chunk_size

        if max_steps is None:
            max_steps = max(np.max(image.size()), np.max(self.mask.size())) * 6
        assert max_steps > 0

        # Prepare image
        image = image.astype(np.bool8)
        for i in self.mask.chunks.keys():
            image.ensure_chunk_at_index(i)

        tasks = [FloodFillTask(i, image=c.to_array()) for i, c in image.chunks.items() if c.any()]

        if not tasks:
            return image

        return self._iterate_fill(image, tasks, max_steps)

    def fill_at_pos(self, position: Vec3i, *args, **kwargs) -> ChunkGrid[bool]:
        image: ChunkGrid[np.bool8] = ChunkGrid(self.mask.chunk_size, dtype=np.bool8, fill_value=np.bool8(False))
        image.set_value(position, True)
        return self.fill(image, *args, **kwargs)


def flood_fill(image: ChunkGrid, mask: ChunkGrid, max_steps: Optional[int] = None, verbose=False,
               **kwargs) -> ChunkGrid[np.bool8]:
    return FloodFillOperator(mask, verbose=verbose).fill(image, max_steps, **kwargs)


def flood_fill_at(position: Vec3i, mask: ChunkGrid, max_steps: Optional[int] = None, verbose=False,
                  **kwargs) -> ChunkGrid[np.bool8]:
    image = ChunkGrid(mask.chunk_size, np.bool8, False)
    image[position] = True
    return flood_fill(image, mask, max_steps, verbose=verbose, **kwargs)
