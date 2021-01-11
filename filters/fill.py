import collections
import functools
import sys
from typing import Optional, List, Tuple, Collection, Set

import numpy as np
from scipy import ndimage

from data.chunks import Index, Chunk, ChunkGrid
from data.faces import ChunkFace
from mathlib import Vec3i

b8 = np.bool8
b8_True = b8(True)
b8_False = b8(False)


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

    def any(self):
        if self._face is not None:
            return True
        elif self._image is not None:
            return bool(np.any(self._image))
        return False

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
        self.mask: ChunkGrid[b8] = mask.copy(dtype=b8)
        self.mask.cleanup(remove=True).pad_chunks(1)
        self.verbose = verbose
        self.__mask_chunks_get = self.mask.chunks.get
        self.min, self.max = self.mask.chunks.minmax()
        self.min -= 1
        self.max += 1

    def is_out_of_bounds(self, index: Index):
        return np.any(index < self.min) or np.any(self.max < index)

    def _compute_fill_task(self, task: FloodFillTask) -> Optional[Tuple[Chunk[b8], List[FloodFillTask]]]:
        """
        Create a fill mask for a chunk with a source mask
        :param task: fill task
        :return: A tuple of the resulting fill mask for this chunk and tasks for the neighboring chunks to fill
        """

        # Do nothing when out of bounds
        if self.is_out_of_bounds(task.index):
            return None

        # Get mask
        mask_chunk: Chunk[b8] = self.mask.ensure_chunk_at_index(task.index, insert=False)

        # Method cache (prevent lookup in loop)
        __face_flip = ChunkFace.flip
        __face_slice = ChunkFace.slice

        if mask_chunk.is_filled():
            if mask_chunk.value:
                return (
                    Chunk(task.index, mask_chunk.size, b8, b8_False).set_fill(b8_True),
                    [
                        FloodFillTask(i, face=__face_flip(f))
                        for f, i in ChunkGrid.iter_neighbors_indices(mask_chunk.index)
                        if f != task.face
                    ]
                )
            else:
                return Chunk(task.index, mask_chunk.size, b8, b8_False), []

        task_image: np.ndarray = task.image(mask_chunk.array_shape)
        assert task_image.shape == mask_chunk.array_shape
        if not task_image.any():
            # return Chunk(task.index, chunk.size, dtype=np_bool8).set_fill(False), []
            return None
        else:
            mask = mask_chunk.to_array()
            # Enforce that only "free" fields in dst_mask are filled
            img = task_image & mask
            result = ndimage.binary_propagation(img, mask=mask).astype(b8)

            # Create tasks where propagation is possible
            tasks = []
            for f, i in ChunkGrid.iter_neighbors_indices(mask_chunk.index):
                face_slice = result[__face_slice(f)]
                if face_slice.all():
                    if f != task.face:
                        tasks.append(FloodFillTask(i, face=__face_flip(f)))
                elif face_slice.any():
                    tmp = np.full(mask_chunk.shape, b8_False, dtype=b8)
                    tmp[__face_slice(__face_flip(f))] = face_slice
                    tasks.append(FloodFillTask(i, image=tmp))
            return Chunk(task.index, mask_chunk.size, b8, b8_False).set_array(result).cleanup(), tasks

    def _fill_fast(self, image: ChunkGrid[b8], tasks: List[FloodFillTask]):
        queue = collections.deque(tasks)
        rest: List[FloodFillTask] = []
        visited: Set[Index] = set()
        visited_filled: Set[Index] = set()

        # Method cache (prevent lookup in loop)
        __image_chunks_get = image.chunks.get
        __iter_neighbors_indicies = ChunkGrid.iter_neighbors_indices
        __face_flip = ChunkFace.flip
        __queue_popleft = queue.popleft
        __queue_extend = queue.extend
        __rest_append = rest.append
        __chunk_is_filled = Chunk.is_filled
        __chunk_set_fill = Chunk.set_fill
        __mask_ensure_chunk_at_index = self.mask.ensure_chunk_at_index

        # Value cache
        __mask_fill_value = bool(self.mask._fill_value)

        def add_visited(value) -> bool:
            n = len(visited)
            visited.add(value)
            return n != len(visited)

        while queue:
            t: FloodFillTask = __queue_popleft()

            if self.is_out_of_bounds(t.index):
                # Found a task outside of bounds
                if __mask_fill_value:
                    image._fill_value = True
                continue

            if add_visited(t.index):
                m: Chunk = __mask_ensure_chunk_at_index(t.index, insert=False)
                if __chunk_is_filled(m) and m.value:
                    visited_filled.add(t.index)
                    # Found a task that can be skipped by filling fast
                    chunk = image.chunks.get(t.index, None)
                    assert chunk is not None
                    chunk.set_fill(b8_True)
                    __queue_extend([
                        FloodFillTask(i, face=__face_flip(f)) for f, i in
                        __iter_neighbors_indicies(t.index)
                        if t.face != f
                    ])
                    continue

            # If task was not handled, add it for further computation
            if t.index not in visited_filled:
                __rest_append(t)
        return rest

    def _iterate_fill(self, image: ChunkGrid[b8], tasks: List[FloodFillTask], max_steps: int) \
            -> ChunkGrid[b8]:

        # Chunk merging function that returns True if it changed
        def merge_chunk_into(src: Chunk[b8], dst: Chunk[b8]) -> bool:
            old = np.copy(dst.value)
            dst |= src
            dst.cleanup()
            return np.any(old != dst.value)

        # Method cache (prevent lookup in loop)
        __list_append = list.append
        __task_merge_all = FloodFillTask.merge_all
        __compute_fill_task = self._compute_fill_task
        __image_chunks_get = image.chunks.get

        # Value cache
        __mask_fill_value = bool(self.mask._fill_value)

        chunk_shape = image.chunk_shape
        for step in range(1, max_steps + 1):
            if step == max_steps:
                if self.verbose:
                    print("Maximum fill mask steps reached!", sys.stderr)
                break

            # Try to fill empty chunks first where no flood filling is needed
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
                # If task is out of bounds, then set the fill_value
                if self.is_out_of_bounds(t.index):
                    if __mask_fill_value:
                        image._fill_value = True
                    continue

                res = __compute_fill_task(t)
                # res is None when there was nothing to do
                # ... or out of bounds, but we already checked that.
                if res is not None:
                    res_chunk, res_tasks = res  # type: Chunk[b8], List[FloodFillTask]
                    index = res_chunk.index

                    # Try to get the chunk at index
                    dst = __image_chunks_get(index, None)
                    assert dst is not None  # We filled the inner bounds already, so the chunk must exist
                    # Merge fill into destination
                    changed = merge_chunk_into(res_chunk, dst)
                    if changed:  # If the merge modified the result, then schedule the next tasks
                        tasks.extend(t for t in res_tasks if t.index in self.mask.chunks)
            if not tasks:
                break
        return image

    def fill(self, image: ChunkGrid[b8], max_steps: Optional[int] = None) -> ChunkGrid[b8]:
        """
        Flood fill the image where it is true using the mask of this fill operator.
        :param image: The starting image for the flood fill
        :param max_steps: maximum propagation steps between chunks
        :return: a filled image
        """
        assert image.chunk_size == self.mask.chunk_size

        if max_steps is None:
            max_steps = max(np.max(image.size()), np.max(self.mask.size())) * 6
        assert max_steps > 0

        # Prepare image
        image = image.copy(dtype=b8, fill_value=b8_False)
        # Fill box of the bounding box, when outside _fill_value is set to true.
        for i in np.ndindex(tuple(self.max - self.min + 1)):
            index = self.min + i
            image.ensure_chunk_at_index(index)

        # Initial fill tasks
        tasks = [FloodFillTask(i, image=c.to_array()) for i, c in image.chunks.items() if c.any()]
        if not tasks:
            return image

        # Dynamic iteration on the tasks which will create new tasks until everything is filled
        result = self._iterate_fill(image, tasks, max_steps)
        if result._fill_value:
            for c in result.chunks:
                c._fill_value = result._fill_value
        return result

    def fill_at_pos(self, position: Vec3i, max_steps: Optional[int] = None) -> ChunkGrid[bool]:
        """
        Start the flood fill at a position
        :param position: the starting point
        :param max_steps: maximum propagation steps between chunks
        :return:
        """
        image: ChunkGrid[b8] = ChunkGrid(self.mask.chunk_size, dtype=b8, fill_value=b8(False))
        image.set_value(position, True)
        return self.fill(image, max_steps=max_steps)


def flood_fill(image: ChunkGrid, mask: ChunkGrid, max_steps: Optional[int] = None, verbose=False,
               **kwargs) -> ChunkGrid[b8]:
    return FloodFillOperator(mask, verbose=verbose).fill(image, max_steps, **kwargs)


def flood_fill_at(position: Vec3i, mask: ChunkGrid, max_steps: Optional[int] = None, verbose=False,
                  **kwargs) -> ChunkGrid[b8]:
    image = ChunkGrid(mask.chunk_size, b8, False)
    image[position] = True
    return flood_fill(image, mask, max_steps, verbose=verbose, **kwargs)
