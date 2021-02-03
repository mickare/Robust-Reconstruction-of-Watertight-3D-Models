from typing import Union, Tuple, Iterator, Optional, Iterable, Callable

import numpy as np

from mathlib import Vec3i

Index = Tuple[int, int, int]
SliceOpt = Union[int, slice, None]


def to_slice(s: SliceOpt = None) -> slice:
    if isinstance(s, slice):
        return s
    if s is None:
        return slice(s)
    return slice(s, s + 1)


def ceildiv(a, b):
    return -(-a // b)


class ValueIter:
    """1D Slice Iterator"""

    @classmethod
    def _indices(cls, s: slice, low: int, high: int, clip=True) -> Tuple[int, int, int]:
        step = s.step or 1
        start = low if s.start is None else s.start
        stop = high if s.stop is None else s.stop
        if clip:
            start = max(start, low + (start - low) % step)
            stop = min(stop, high)
        else:
            start = start
            stop = stop
        return start, stop, step

    def __init__(self, s: SliceOpt, low: int, high: int, clip=True):
        self._low = int(low)
        self._high = int(high)
        self._slice = to_slice(s)
        self._start, self._stop, self._step = self._indices(self._slice, low, high, clip)
        self.clip = clip

    def range(self) -> range:
        return range(self._start, self._stop, self._step)

    def __contains__(self, item) -> bool:
        if isinstance(item, int):
            return self._start <= item < self._stop and (item % self._step) == (self._start % self._step)
        return False

    def __iter__(self) -> Iterator[int]:
        yield from range(self._start, self._stop, self._step)

    def __len__(self) -> int:
        ds = self._stop - self._start
        return max(0, ds // self._step, ds % self._step > 0)

    @property
    def low(self) -> int:
        return self._low

    @property
    def high(self) -> int:
        return self._high

    @property
    def slice(self) -> slice:
        return self._slice

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def step(self) -> int:
        return self._step

    def __floordiv__(self, other):
        return ValueIter(
            slice(
                self._start // other,
                ceildiv(self._stop, other),
                max(1, self._step // other)
            ),
            self._low // other,
            ceildiv(self._high, other),
            self.clip
        )


class PositionIter:
    """3D Slice Iterator"""

    @classmethod
    def require_bounded(cls, x: SliceOpt, y: SliceOpt, z: SliceOpt) -> "PositionIter":
        x = to_slice(x)
        y = to_slice(y)
        z = to_slice(z)
        assert x.start is not None and x.stop is not None
        assert y.start is not None and y.stop is not None
        assert z.start is not None and z.stop is not None
        return cls(x, y, z, low=(x.start, y.start, z.start), high=(x.stop, y.stop, z.stop))

    @classmethod
    def empty(cls) -> "PositionIter":
        return cls(0, 0, 0, np.zeros(3), np.zeros(3))

    def __init__(self, x: SliceOpt, y: SliceOpt, z: SliceOpt, low: Vec3i, high: Vec3i, clip=True):
        self._low = np.asarray(low, dtype=int)
        self._high = np.asarray(high, dtype=int)
        assert self._low.shape == (3,) and self._high.shape == (3,)
        self._x = ValueIter(x, self._low[0], self._high[0], clip)
        self._y = ValueIter(y, self._low[1], self._high[1], clip)
        self._z = ValueIter(z, self._low[2], self._high[2], clip)
        self.clip = clip

    def __contains__(self, item: Vec3i) -> bool:
        if len(item) == 3:
            return item[0] in self._x and item[1] in self._y and item[2] in self._z
        return False

    def iter_with_indices(self) -> Iterator[Tuple[Index, Index]]:
        for i, u in enumerate(self._x.range()):
            for j, v in enumerate(self._y.range()):
                for k, w in enumerate(self._z.range()):
                    yield (i, j, k), (u, v, w)

    def __iter__(self) -> Iterator[Index]:
        for u in self._x.range():
            for v in self._y.range():
                for w in self._z.range():
                    yield u, v, w

    def __len__(self):
        return len(self._x) * len(self._y) * len(self._z)

    @property
    def shape(self):
        return len(self._x), len(self._y), len(self._z)

    @property
    def low(self) -> Vec3i:
        return self._low

    @property
    def high(self) -> Vec3i:
        return self._high

    @property
    def x(self) -> ValueIter:
        return self._x

    @property
    def y(self) -> ValueIter:
        return self._y

    @property
    def z(self) -> ValueIter:
        return self._z

    @property
    def start(self) -> Vec3i:
        return np.asarray((self._x.start, self._y.start, self._z.start), dtype=int)

    @property
    def stop(self) -> Vec3i:
        return np.asarray((self._x.stop, self._y.stop, self._z.stop), dtype=int)

    @property
    def step(self) -> Vec3i:
        return np.asarray((self._x.step, self._y.step, self._z.step), dtype=int)

    def __floordiv__(self, other):
        x = self._x // other
        y = self._y // other
        z = self._z // other
        return PositionIter(
            x.slice, y.slice, z.slice,
            self._low // other,
            ceildiv(self._high, other),
            self.clip
        )


class MinMaxCheck:
    """3D minmax check"""
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
        self._min = tuple(np.min(indices, axis=0))
        self._max = tuple(np.max(indices, axis=0))
        self._dirty = False

    def add(self, index: Index):
        if self._min is None:
            self._min = tuple(np.asarray(index, dtype=int))
        else:
            self._min = tuple(np.min((self._min, index), axis=0))
        if self._max is None:
            self._max = tuple(np.asarray(index, dtype=int))
        else:
            self._max = tuple(np.max((self._max, index), axis=0))

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
