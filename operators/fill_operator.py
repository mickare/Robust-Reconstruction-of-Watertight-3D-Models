import collections
import functools
import multiprocessing
import os
import sys
from typing import Optional, Generic, List, TypeVar, Tuple, Collection, Dict

import numpy as np
import tqdm

from data.chunks import Index, ChunkFace, Chunk, ChunkType, ChunkGrid
from mathlib import Vec3i

T = TypeVar('T')


class BleedingFillTask(Generic[T]):
    __slots__ = ["index", "full", "mask"]

    def __init__(self, index: Index, full: ChunkFace = None, mask: Optional[np.ndarray] = None):
        assert len(index) == 3
        self.index = tuple(index)
        self.full = full
        self.mask = mask

    def merge(self, other: "BleedingFillTask", grid: ChunkGrid):
        assert self.index == other.index
        if self.full is not None:
            if other.full is not None:
                if self.full == other.full:
                    return BleedingFillTask(self.index, self.full)
        return BleedingFillTask(self.index, mask=self.to_mask(grid) | other.to_mask(grid))

    def to_mask(self, grid: ChunkGrid) -> np.ndarray:
        if self.full:
            m = grid.empty_mask(False)
            m[self.full.slice()] = True
            return m
        elif self.mask is not None:
            return self.mask
        else:
            return grid.empty_mask(False)

    def __eq__(self, other):
        if isinstance(other, BleedingFillTask):
            if self.index == other.index:
                return self.full == other.full and np.all(self.mask == other.mask)
        return False

    @classmethod
    def merge_all(cls, tasks: Collection["BleedingFillTask"], grid: ChunkGrid) -> "BleedingFillTask":
        def _merge(a: "BleedingFillTask", b: "BleedingFillTask"):
            return a.merge(b, grid)

        return functools.reduce(_merge, tasks)


class FillMask:
    __slots__ = ["index", "full", "mask"]

    def __init__(self, index: Index, full: bool = False, mask: Optional[np.ndarray] = None):
        assert len(index) == 3
        self.index = tuple(index)
        self.full = full
        self.mask = mask

    def merge(self, other: "FillMask") -> bool:
        """
        Merge another fill mask into this one.
        :param other: FillMask merging into this one
        :return: True if this mask changed, else False
        """
        assert self.index == other.index
        if other.full:
            changed = self.full != other.full
            self.full = other.full
            self.mask = None
            return changed
        if self.full:
            return False
        if other.mask is not None:
            if self.mask is None:
                self.mask = other.mask
                return True
            else:
                changed = (self.mask != other.mask).any()
                self.mask |= other.mask
                return changed
        return False

    def apply(self, chunk: Chunk[T], value: T):
        if self.full:
            chunk.set_fill(value)
        elif self.mask is not None:
            tmp = chunk.to_array()
            tmp[self.mask] = value
            chunk.set_array(tmp)


class FillOperator(Generic[T]):

    def __init__(self, grid: ChunkGrid[T], replace=False, replace_value: Optional[T] = None):
        self.grid = grid
        self.replace = replace
        self.replace_value = replace_value

    def _create_fill_mask(self, chunk: Chunk, initial_mask: np.ndarray) -> Tuple[FillMask, List[BleedingFillTask[T]]]:
        """
        Fill the chunk with a value at a position
        :param value:
        :param pos:
        :return: dict of neighboring chunk indicies that are touched by the fill and the positions
        """
        assert initial_mask.shape == chunk.shape
        if not initial_mask.any():
            return FillMask(chunk.index), []
        if chunk.type == ChunkType.EMPTY:
            return (
                FillMask(chunk.index, full=True),
                [BleedingFillTask(i, full=f) for f, i in self.grid.iter_neighbors_indicies(chunk.index)]
            )
        elif chunk.type == ChunkType.FILL:
            if self.replace and chunk.value == self.replace_value:
                return (
                    FillMask(chunk.index, full=True),
                    [BleedingFillTask(i, full=f) for f, i in self.grid.iter_neighbors_indicies(chunk.index)]
                )
            else:
                return FillMask(chunk.index), []
        elif chunk.type == ChunkType.ARRAY:
            arr: np.ndarray = chunk.value
            assert isinstance(arr, np.ndarray)
            if self.replace:
                original_mask = chunk.mask(value=self.replace_value)
            else:
                original_mask = chunk.mask(logic=False)

            if original_mask[initial_mask].any():
                # If any initial mask hit an empty spot we can continue to fill it

                # Create the fill mask
                mask = np.full(chunk.shape, False, dtype=np.bool)
                mask[initial_mask] = True

                # dilate fill mask
                for _ in range(chunk.size + 1):
                    old = mask
                    padded = np.pad(mask, 1)
                    mask = mask | \
                           padded[0:-2, 1:-1, 1:-1] | \
                           padded[2:, 1:-1, 1:-1] | \
                           padded[1:-1, 0:-2, 1:-1] | \
                           padded[1:-1, 2:, 1:-1] | \
                           padded[1:-1, 1:-1, 0:-2] | \
                           padded[1:-1, 1:-1, 2:]
                    mask &= original_mask
                    if (old == mask).all():  # No change
                        break

                tasks = []
                for f, i in self.grid.iter_neighbors_indicies(chunk.index):
                    bleeding = mask[f.slice()]
                    if bleeding.any():
                        tmp = np.full(chunk.shape, False, dtype=np.bool)
                        tmp[f.flip().slice()] = bleeding
                        tasks.append(BleedingFillTask(i, mask=tmp))
                return FillMask(chunk.index, mask=mask), tasks
        else:
            raise RuntimeError(f"Unexpected chunk type {chunk.type}")

    def _call_create_fill_mask(self, task: BleedingFillTask):
        chunk = self.grid.chunks.get(task.index)
        if chunk:
            return self._create_fill_mask(chunk, task.to_mask(self.grid))
        return None

    def fill_masks_parallel(self, start: Vec3i, max_steps=10) -> Dict[Index, FillMask]:
        start_chunk = self.grid.chunk_at_pos(start)
        assert start_chunk  # Must be a valid start chunk

        cpus = os.cpu_count()

        # Initial first chunk
        initial_mask = np.full(self.grid.chunk_shape, False, dtype=np.bool)
        initial_mask[start_chunk.inner(start)] = True
        start_mask, start_tasks = self._create_fill_mask(start_chunk, initial_mask=initial_mask)

        # Results
        tasks = start_tasks
        masks: Dict[Index, FillMask] = {start_mask.index: start_mask}

        # Rest in parallel
        # grid_var = contextvars.ContextVar('_grid')
        # grid_var.set(grid)
        # self_var = contextvars.ContextVar('_fill_operator')
        # self_var.set(self)

        with multiprocessing.Pool(cpus) as pool:

            for step in range(max_steps + 1):
                if step == max_steps:
                    print("Maximum fill mask steps reached!", sys.stderr)
                    break

                # Reduce
                tasks_indexed = collections.defaultdict(list)
                for t in tasks:
                    tasks_indexed[tuple(t.index)].append(t)
                tasks_reduced: List[BleedingFillTask] = [BleedingFillTask.merge_all(ts, self.grid) for ts in
                                                         tasks_indexed.values()]

                # Rebuild tasks from result
                tasks = []
                pbar = tqdm.tqdm(total=len(tasks_reduced), desc=f"Fill step {step}")

                # for result in pool.imap_unordered(self._call_create_fill_mask, tasks_reduced,
                #                                   chunksize=max(1, min(16, len(tasks_reduced) // cpus))):

                for task in tasks_reduced:
                    result = self._call_create_fill_mask(task)

                    pbar.update(1)
                    if result is not None:
                        result_mask, result_tasks = result  # type: FillMask, List[BleedingFillTask]
                        index = result_mask.index
                        if index not in masks:
                            masks[index] = result_mask
                            changed = True
                        else:
                            changed = masks[index].merge(result_mask)
                        if changed:
                            for t in result_tasks:
                                if t.index in self.grid.chunks:
                                    tasks.append(t)
                if not tasks:
                    break

        return masks
