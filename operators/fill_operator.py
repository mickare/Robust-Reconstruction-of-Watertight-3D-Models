import collections
import functools
import multiprocessing
import os
import sys
from typing import Optional, List, Tuple, Collection, Union, Set

import numpy as np
import tqdm
from scipy import ndimage

from data.chunks import Index, ChunkFace, Chunk, ChunkGrid
from mathlib import Vec3i


class FloodFillTask:
    __slots__ = ["index", "_face", "_image"]

    def __init__(self, index: Index, face: ChunkFace = None, image: Optional[np.ndarray] = None):
        assert len(index) == 3
        self.index = tuple(index)
        self._face = face
        self._image = image
        if face is not None:
            assert isinstance(face, ChunkFace)
        if image is not None:
            assert isinstance(image, np.ndarray)

    def merge(self, other: "FloodFillTask", shape: Tuple[int, ...]):
        assert self.index == other.index
        if self._face is not None:
            if other._face is not None:
                if self._face == other._face:
                    return FloodFillTask(self.index, self._face)
        mask = self.image(shape) | other.image(shape)
        return FloodFillTask(self.index, image=mask)

    def image(self, shape: Tuple[int, ...]) -> np.ndarray:
        if self._face:
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

    @classmethod
    def merge_all(cls, tasks: Collection["FloodFillTask"], shape: Tuple[int, ...]) -> "FloodFillTask":
        def _merge(a: "FloodFillTask", b: "FloodFillTask"):
            return a.merge(b, shape)

        return functools.reduce(_merge, tasks)


class FloodFillOperator:

    def __init__(self, mask: ChunkGrid[bool], verbose=False):
        self.mask = mask.astype(bool)
        self.verbose = verbose

    def _compute_fill_task(self, task: FloodFillTask) -> Optional[Tuple[Chunk[bool], List[FloodFillTask]]]:
        """
        Create a fill mask for a chunk with a source mask
        :param task: fill task
        :return: A tuple of the resulting fill mask for this chunk and tasks for the neighboring chunks to fill
        """
        chunk: Chunk[bool] = self.mask.chunks.get(task.index)
        if chunk is None:
            return None

        image: np.ndarray = task.image(chunk.shape)

        assert image.shape == chunk.shape
        if not image.any():
            return Chunk(task.index, chunk.size, dtype=bool).set_fill(False), []

        if chunk.is_filled():
            if chunk.value:
                return (
                    Chunk(task.index, chunk.size, dtype=bool).set_fill(True),
                    [FloodFillTask(i, face=f) for f, i in self.mask.iter_neighbors_indicies(chunk.index)]
                )
            else:
                return Chunk(task.index, chunk.size, dtype=bool).set_fill(False), []
        else:
            mask = chunk.to_array()

            # Is there any chance it can be filled?
            if mask.any():
                # Enforce that only "free" fields in dst_mask are filled
                img = image & mask
                result = ndimage.binary_propagation(img, mask=mask).astype(np.bool)

                tasks = []
                for f, i in self.mask.iter_neighbors_indicies(chunk.index):
                    face_slice = result[f.slice()]
                    if face_slice.all():
                        tasks.append(FloodFillTask(i, face=f))
                    elif face_slice.any():
                        tmp = np.full(chunk.shape, False, dtype=np.bool)
                        tmp[f.flip().slice()] = face_slice
                        tasks.append(FloodFillTask(i, image=tmp))
                return Chunk(task.index, chunk.size, dtype=bool).set_array(result), tasks

    def _fill_fast(self, image: ChunkGrid[bool], tasks: List[FloodFillTask]):
        queue = collections.deque(tasks)
        remainder: List[FloodFillTask] = []
        visited: Set[Index] = set()
        while queue:
            t = queue.popleft()
            if t.index in visited:
                continue
            visited.add(t.index)

            m: Chunk = self.mask.chunks.get(t.index, None)
            if m is not None:
                if m.is_filled():
                    if m.value:
                        o = image.chunks.get(t.index)
                        if o is not None:
                            if o.is_filled():
                                if not o.value:
                                    o.set_fill(True)
                                    queue.extend(
                                        [FloodFillTask(i, face=f) for f, i in
                                         self.mask.iter_neighbors_indicies(t.index)]
                                    )
                            else:
                                remainder.append(t)
                else:
                    remainder.append(t)
        return remainder

    @classmethod
    def _merge_chunk_into(cls, src: Chunk[bool], dst: Chunk[bool]) -> bool:
        old = np.copy(dst.value)
        dst.ior(src)
        return np.any(old != dst.value)

    def _iterate_fill(self, image: ChunkGrid[bool], tasks: List[FloodFillTask], max_steps: int,
                      pool: Optional[multiprocessing.Pool] = None, workers: int = 1, **kwargs):
        chunk_shape = self.mask.chunk_shape
        for step in range(max_steps + 1):
            if step == max_steps:
                print("Maximum fill mask steps reached!", sys.stderr)
                break

            # First do fast fill operations (e.g. empty chunk)
            tasks = self._fill_fast(image, tasks)

            # Reduce and Merge tasks
            tasks_indexed = collections.defaultdict(list)
            for t in tasks:
                tasks_indexed[tuple(t.index)].append(t)
            tasks_reduced: List[FloodFillTask] = [FloodFillTask.merge_all(ts, chunk_shape) for ts in
                                                  tasks_indexed.values()]

            # Rebuild tasks for next iteration
            tasks = []
            pbar = None
            if self.verbose:
                pbar = tqdm.tqdm(total=len(tasks_reduced), desc=f"Fill step {step}")
            if pool is not None:
                iterator = pool.imap_unordered(self._compute_fill_task, tasks_reduced,
                                               chunksize=max(1, min(32, len(tasks_reduced) // workers)))
            else:
                iterator = (self._compute_fill_task(t) for t in tasks_reduced)

            for res in iterator:
                if self.verbose:
                    pbar.update(1)
                if res is not None:
                    res_mask, res_tasks = res  # type: Chunk, List[FloodFillTask]
                    index = res_mask.index
                    o = image.chunks.get(index)
                    if o is not None:
                        changed = self._merge_chunk_into(res_mask, image.chunks[index])
                        if changed:
                            for t in res_tasks:
                                if t.index in self.mask.chunks:
                                    tasks.append(t)
            if not tasks:
                break
        return image

    def _fill(self, image: ChunkGrid[bool], tasks: List[FloodFillTask], max_steps=1000, workers: int = 0,
              **kwargs) -> ChunkGrid[bool]:
        assert max_steps > 0
        assert image.chunk_size == self.mask.chunk_size
        # Results
        tasks = list(tasks)

        if not tasks:
            return image

        if workers > 1:
            with multiprocessing.Pool(workers) as pool:
                return self._iterate_fill(image, tasks, max_steps, pool=pool, workers=workers, **kwargs)
        else:
            return self._iterate_fill(image, tasks, max_steps)

    def fill(self, image: ChunkGrid, max_steps: Optional[int] = None, workers: Union[bool, int] = False, **kwargs) \
            -> ChunkGrid[bool]:

        if max_steps is None:
            max_steps = max(np.max(image.size()), np.max(self.mask.size())) * 6

        if isinstance(workers, bool) and workers:
            workers = os.cpu_count()

        # Prepare image
        image = image.astype(bool)
        for i in self.mask.chunks.keys():
            image.ensure_chunk_at_index(i)

        tasks = [FloodFillTask(i, image=c.to_array()) for i, c in image.chunks.items() if c.any()]
        return self._fill(image, tasks, max_steps=max_steps, workers=int(workers))

    def fill_at_pos(self, position: Vec3i, *args, **kwargs) -> ChunkGrid[bool]:
        image = ChunkGrid(self.mask.chunk_size, bool, False)
        image.set_pos(position, True)
        return self.fill(image, *args, **kwargs)


def flood_fill(image: ChunkGrid, mask: ChunkGrid,
               max_steps: Optional[int] = None, workers: Union[bool, int] = False, verbose=False,
               **kwargs) -> ChunkGrid[bool]:
    return FloodFillOperator(mask, verbose=verbose).fill(image, max_steps, workers, **kwargs)


def flood_fill_at(position: Vec3i, mask: ChunkGrid,
                  max_steps: Optional[int] = None, workers: Union[bool, int] = False, verbose=False,
                  **kwargs) -> ChunkGrid[bool]:
    image = ChunkGrid(mask.chunk_size, bool, False)
    image.set_pos(position, True)
    return flood_fill(image, mask, max_steps, workers, verbose=verbose, **kwargs)
