from typing import Union, Tuple, Iterator, Optional, Iterable, Callable

import numpy as np

from data.chunks import Index


class SliceIterator:
    def __init__(self, s: Union[slice, int, None], bound: Tuple[int, int]):
        self._s = s
        self.min = np.min(bound)
        self.max = np.max(bound)
        self.start = self.min
        self.stop = self.max + 1
        self.mod_step = 1
        self.mod_value = 0

        if s is not None:
            if isinstance(s, int):
                self.start = s
                self.stop = s + 1
            elif isinstance(s, slice):
                if s.start is not None:
                    if s.start < 0:
                        self.start = self.max + 1 + s.start
                    else:
                        self.start = s.start

                if s.start is not None:
                    if s.stop < 0:
                        self.stop = self.max + 1 + s.stop
                    else:
                        self.stop = s.stop

                if s.step is not None:
                    self.mod_step = s.step
                    if s.start is not None:
                        self.mod_value = s.start % self.mod_step

    def __contains__(self, item):
        if isinstance(item, int):
            return self.start <= item < self.stop and item & self.mod_step == self.mod_value
        return False

    def __iter__(self) -> Iterator[int]:
        yield from range(self.start, self.stop, self.mod_step)

    def __len__(self):
        ds = self.stop - self.start
        return max(0, ds // self.mod_step, ds % self.mod_step > 0)


class MinMaxCheck:
    __slots__ = ["_min", "_max", "_dirty"]

    def __init__(self):
        self._min: Optional[Index] = None
        self._max: Optional[Index] = None
        self._dirty = False

    def clear(self):
        self._min = None
        self._max = None
        self._dirty = False

    def update(self, other: Union[Iterable[Index]]):
        indices = np.array(list(other))
        assert len(indices) > 0
        self._min = np.min(indices, axis=0)
        self._max = np.max(indices, axis=0)
        self._dirty = False

    def add(self, index: Index):
        if self._min is None:
            self._min = index
        else:
            self._min = np.min((self._min, index), axis=0)
        if self._max is None:
            self._max = index
        else:
            self._max = np.max((self._max, index), axis=0)

    @property
    def dirty(self):
        return self._dirty

    def set_dirty(self):
        self._dirty = True

    @property
    def min(self):
        assert not self._dirty
        return self._min

    @property
    def max(self):
        assert not self._dirty
        return self._max

    def get(self) -> Tuple[Index, Index]:
        assert not self._dirty
        return self._min, self._max

    def safe(self, getter: Callable[[], Union[Iterable[Index]]]) -> Tuple[Index, Index]:
        if self._dirty:
            self.update(getter())
        return self.get()