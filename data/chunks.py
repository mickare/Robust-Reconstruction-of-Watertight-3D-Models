import enum
import functools
import itertools
import operator
from typing import Union, Tuple, Iterator, Optional, Generic, TypeVar, Callable, Type, Sequence, Set

import numpy as np
import sparse

from data.data_utils import PositionIter
from data.index_dict import IndexDict, Index
from mathlib import Vec3i

V = TypeVar('V')
M = TypeVar('M')
ChunkIndex = Index


class ChunkFace(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    TOP = 2
    BOTTOM = 3
    EAST = 4
    WEST = 5

    @property
    def direction(self):
        return ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))[self]

    def flip(self) -> "ChunkFace":
        return ChunkFace((self // 2) * 2 + ((self + 1) % 2))

    def slice(self, width: int = -1) -> Tuple[Union[int, slice], ...]:
        s = slice(None)
        s0, s1 = -1, 0
        if width >= 0:
            s0 = slice(-1 - width, -1)
            s1 = slice(0, width)
        return ((s0, s, s),
                (s1, s, s),
                (s, s0, s),
                (s, s1, s),
                (s, s, s0),
                (s, s, s1))[self]

    def shape(self, size: int) -> Tuple[int, int, int]:
        return ((1, size, size),
                (1, size, size),
                (size, 1, size),
                (size, 1, size),
                (size, size, 1),
                (size, size, 1))[self]

    def __bool__(self):
        return True

    @classmethod
    def corners(cls) -> Iterator[Tuple["ChunkFace", "ChunkFace", "ChunkFace"]]:
        return itertools.product((ChunkFace.NORTH, ChunkFace.SOUTH),
                                 (ChunkFace.TOP, ChunkFace.BOTTOM),
                                 (ChunkFace.EAST, ChunkFace.WEST))

    @classmethod
    def corner_slice(cls, x: "ChunkFace", y: "ChunkFace", z: "ChunkFace", width: int = -1) \
            -> Tuple[Union[int, slice], ...]:
        s = slice(None)
        s0, s1 = -1, 0
        if width >= 0:
            s0 = slice(-1 - width, -1)
            s1 = slice(0, width)
        u = x % 2 == 0
        v = y % 2 == 0
        w = z % 2 == 0
        return (
            s0 if u else s1,
            s0 if v else s1,
            s0 if w else s1
        )

    @classmethod
    def corner_direction(cls, x: "ChunkFace", y: "ChunkFace", z: "ChunkFace") -> Vec3i:
        return np.add(x.direction, y.direction) + z.direction


class ChunkHelper:
    class _IndexMeshGrid:
        def __getitem__(self, item) -> Iterator[Vec3i]:
            assert isinstance(item, slice)
            xs, ys, zs = np.mgrid[item, item, item]
            return zip(xs.flatten(), ys.flatten(), zs.flatten())

    indexGrid = _IndexMeshGrid()


class BetterPartialMethod(functools.partialmethod):
    def __init__(self, func,
                 *args,
                 op: Callable[[Union[np.ndarray, V], Union[np.ndarray, V]], Union[np.ndarray, M]],
                 desc=None, **kwargs) -> None:
        super(BetterPartialMethod, self).__init__(func, *args, op=op, **kwargs)
        self.__doc__ = getattr(op, "__doc__", None) if desc is None else desc

    def __get__(self, instance, owner=None):
        f = super(BetterPartialMethod, self).__get__(instance, owner)
        f.__doc__ = self.__doc__
        return f


class Chunk(Generic[V]):
    def __init__(self, index: Vec3i, size: int, dtype: Optional[Type[V]] = None, fill_value: Optional[V] = None):
        self._index: Vec3i = np.asarray(index, dtype=np.int)
        self._size = size
        self._dtype = np.dtype(dtype).type
        self._fill_value = self._dtype() if fill_value is None else self._dtype(fill_value)
        self._is_filled = True
        self._value: Union[V, np.ndarray] = self._fill_value

    @property
    def index(self) -> Vec3i:
        return self._index

    @property
    def value(self) -> Union[None, V, np.ndarray]:
        return self._value

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._size, self._size, self._size

    @property
    def position_low(self) -> Vec3i:
        return np.multiply(self._index, self._size)

    @property
    def position_high(self) -> Vec3i:
        return self.position_low + self._size

    def is_filled(self) -> bool:
        return self._is_filled

    def is_array(self) -> bool:
        return not self._is_filled

    def inner(self, pos: Vec3i) -> np.ndarray:
        return np.asarray(pos, dtype=np.int) % self._size

    def get_pos(self, pos: Vec3i) -> V:
        return self.to_array()[tuple(self.inner(pos))]

    def set_pos(self, pos: Vec3i, value: V):
        inner = self.inner(pos)
        arr = self.to_array()
        arr[tuple(inner)] = value
        self.set_array(arr)

    def set_or_fill(self, pos: Vec3i, value: V):
        if self._is_filled:
            self.set_fill(value)
        else:
            self.set_pos(pos, value)

    def set_fill(self, value: V) -> "Chunk[V]":
        self._value = value
        self._is_filled = True
        return self

    def set_array(self, value: np.ndarray) -> "Chunk[V]":
        assert self.shape == value.shape, f"{self.shape} != {value.shape}"
        self._value = np.array(value, dtype=self._dtype)
        self._is_filled = False
        return self

    def to_array(self) -> np.ndarray:
        if self._is_filled:
            return np.full(self.shape, self._value, dtype=self._dtype)
        else:
            return self._value

    def where(self, other: "Chunk") -> "Chunk[V]":
        other: Chunk[bool] = other.astype(bool)
        c = self.copy(empty=True)
        if other.is_filled() and other._value:
            c.set_fill(self._value)
        else:
            arr = np.full(self.shape, self._fill_value, dtype=self._dtype)
            arr[other._value] = self._value[other._value]
            c.set_array(arr)
        return c

    def __getitem__(self, item):
        if isinstance(item, Chunk):
            return self.where(item)
        return self.to_array()[item]

    def __setitem__(self, key: Union[np.ndarray, "Chunk"], value: Union[V, np.ndarray, "Chunk"]):
        if isinstance(key, Chunk):
            key = key.astype(bool).to_array()
        if isinstance(value, Chunk):
            value = value.to_array()
        arr = self.to_array()
        arr[key] = value
        self.set_array(arr)
        self.cleanup_memory()

    def copy(self, empty=False, dtype=None):
        dtype = dtype or self._dtype
        c = Chunk(self._index, self._size, dtype=dtype, fill_value=dtype(self._fill_value))
        if not empty:
            if self.is_filled():
                c.set_fill(self._value)
            else:
                c.set_array(self._value.copy())
        return c

    def split(self, splits: int, chunk_size: Optional[int] = None) -> Iterator["Chunk"]:
        assert splits > 0 and self._size % splits == 0
        splits = int(splits)
        split_size = self._size // splits

        # New chunk size
        chunk_size = int(chunk_size or self._size)
        assert chunk_size > 0
        # Voxel repeats to achieve chunk size
        repeats = chunk_size / split_size
        assert repeats > 0 and repeats % 1 == 0  # Chunk size must be achieved by duplication of voxels
        repeats = int(repeats)

        for offset in ChunkHelper.indexGrid[:splits]:
            new_index = np.add(self._index * splits, offset)
            c = Chunk(new_index, size=chunk_size, fill_value=self._fill_value)
            if self.is_filled():
                c.set_fill(self._value)
            else:
                u, v, w = np.asarray(offset, dtype=np.int) * split_size
                tmp = self._value[u: u + split_size, v: v + split_size, w: w + split_size]
                if repeats == 1:
                    val = tmp.copy()
                else:
                    val = np.repeat(np.repeat(np.repeat(tmp, repeats, axis=0), repeats, axis=1), repeats, axis=2)
                c.set_array(val)
            yield c

    def convert(self, func: Callable[[V], M], func_vec: Optional[Callable[[np.ndarray], np.ndarray]]) -> "Chunk[M]":
        func_vec = func_vec or np.vectorize(func)
        c = Chunk(self._index, self._size, fill_value=func(self._fill_value))
        if self.is_filled():
            c.set_fill(func(self._value))
        else:
            c.set_array(func_vec(self._value))
        return c

    def astype(self, dtype: Type[M]) -> "Chunk[M]":
        if self._dtype == dtype:
            return self
        c = Chunk(self._index, self._size, dtype=dtype, fill_value=dtype(self._fill_value))
        if self.is_filled():
            c.set_fill(dtype(self._value))
        else:
            c.set_array(self._value.astype(dtype))
        return c

    def __bool__(self):
        raise RuntimeWarning(f"Do not use __bool__ on {self.__class__}")

    def cleanup_memory(self):
        """Try to reduce memory footprint"""
        if self.is_array():
            if self._dtype == bool:
                if np.all(self._value):
                    self.set_fill(True)
                elif not np.any(self._value):
                    self.set_fill(False)
            else:
                u = np.unique(self._value)
                if len(u) == 1:
                    self.set_fill(u.item())
        return self

    def all(self) -> bool:
        return np.all(self._value)

    def any(self) -> bool:
        return np.any(self._value)

    def padding(self, grid: "ChunkGrid[V]", padding: int, corners=False) -> np.ndarray:
        arr = np.pad(self.to_array(), padding)
        for face, index in grid.iter_neighbors_indicies(self._index):
            c = grid.ensure_chunk_at_index(index, insert=False)
            arr[face.slice(padding)] = np.pad(c.to_array(), padding)[face.flip().slice(padding)]
        if corners:
            for u, v, w in ChunkFace.corners():
                s0 = ChunkFace.corner_slice(u, v, w, width=padding)
                s1 = ChunkFace.corner_slice(u.flip(), v.flip(), w.flip(), width=padding)
                d = ChunkFace.corner_direction(u, v, w)
                arr[s0] = grid.ensure_chunk_at_index(d + self._index, insert=False).to_array()[s1]
        return arr

    def invert(self, inplace=False) -> "Chunk[V]":
        c = self if inplace else self.copy(empty=True)
        c._value = np.invert(self._value)
        return c

    def equals(self, other: Union["Chunk", V]) -> "Chunk[bool]":
        # return self._op(other, operator.eq, bool)
        return self.eq(other)

    def _operator1(self,
                   op: Callable[[Union[np.ndarray, V]], Union[np.ndarray, M]],
                   dtype: Optional[Type[M]] = None, inplace=False, ) -> "Chunk[M]":
        dtype = dtype or self.dtype
        # Inplace selection
        c = self if inplace else self.copy()
        c._value = op(self._value)
        if dtype is not None:
            c._dtype = dtype
            c._fill_value = dtype(self._fill_value)
        return c

    def _operator2(self, other: Union["Chunk[V]", np.ndarray, V],
                   op: Callable[[Union[np.ndarray, V], Union[np.ndarray, V]], Union[np.ndarray, M]],
                   dtype: Optional[Type[M]] = None, inplace=False, ) -> "Chunk[M]":
        dtype = dtype or self.dtype
        # Inplace selection
        c = self if inplace else self.copy()

        if isinstance(other, Chunk):
            c._is_filled = c._is_filled & other._is_filled
            c._value = op(c._value, other._value)
        else:
            c._is_filled = c._is_filled
            c._value = op(c._value, other)

        # Update dtype and if possible the default fill value
        # noinspection DuplicatedCode
        if dtype is not None:
            c._dtype = dtype
            try:
                if isinstance(other, Chunk):
                    c._fill_value = dtype(op(self._fill_value, other._fill_value))
                else:
                    c._fill_value = dtype(op(self._fill_value, other))
            except Exception:
                try:
                    c._fill_value = dtype(self._fill_value)
                except Exception:
                    pass
        c.cleanup_memory()
        return c

    # Operators
    __eq__ = BetterPartialMethod(_operator2, op=operator.eq, dtype=bool)  # type: Chunk[bool]
    __ne__ = BetterPartialMethod(_operator2, op=operator.ne, dtype=bool)  # type: Chunk[bool]
    __lt__ = BetterPartialMethod(_operator2, op=operator.lt, dtype=bool)  # type: Chunk[bool]
    __le__ = BetterPartialMethod(_operator2, op=operator.le, dtype=bool)  # type: Chunk[bool]
    __gt__ = BetterPartialMethod(_operator2, op=operator.gt, dtype=bool)  # type: Chunk[bool]
    __ge__ = BetterPartialMethod(_operator2, op=operator.ge, dtype=bool)  # type: Chunk[bool]

    __abs__ = BetterPartialMethod(_operator1, op=operator.abs)  # type: Chunk[V]
    __invert__ = BetterPartialMethod(_operator1, op=operator.inv)  # type: Chunk[V]
    __neg__ = BetterPartialMethod(_operator1, op=operator.neg)  # type: Chunk[V]

    __add__ = BetterPartialMethod(_operator2, op=operator.add)  # type: Chunk[V]
    __and__ = BetterPartialMethod(_operator2, op=operator.and_)  # type: Chunk[V]
    __floordiv__ = BetterPartialMethod(_operator2, op=operator.floordiv)  # type: Chunk[V]
    __mod__ = BetterPartialMethod(_operator2, op=operator.mod)  # type: Chunk[V]
    __mul__ = BetterPartialMethod(_operator2, op=operator.mul)  # type: Chunk[V]
    __matmul__ = BetterPartialMethod(_operator2, op=operator.matmul)  # type: Chunk[V]
    __or__ = BetterPartialMethod(_operator2, op=operator.or_)  # type: Chunk[V]
    __sub__ = BetterPartialMethod(_operator2, op=operator.sub)  # type: Chunk[V]
    __truediv__ = BetterPartialMethod(_operator2, op=operator.truediv, dtype=np.float)  # type: Chunk[float]
    __xor__ = BetterPartialMethod(_operator2, op=operator.xor)  # type: Chunk[V]

    __iand__ = BetterPartialMethod(_operator2, op=operator.iand, inplace=True)  # type: Chunk[V]
    __ior__ = BetterPartialMethod(_operator2, op=operator.ior, inplace=True)  # type: Chunk[V]
    __ixor__ = BetterPartialMethod(_operator2, op=operator.ixor, inplace=True)  # type: Chunk[V]
    __iadd__ = BetterPartialMethod(_operator2, op=operator.iadd, inplace=True)  # type: Chunk[V]
    __isub__ = BetterPartialMethod(_operator2, op=operator.isub, inplace=True)  # type: Chunk[V]
    __imul__ = BetterPartialMethod(_operator2, op=operator.imul, inplace=True)  # type: Chunk[V]
    __itruediv__ = BetterPartialMethod(_operator2, op=operator.itruediv, dtype=np.float,
                                       inplace=True)  # type: Chunk[float]
    __ifloordiv__ = BetterPartialMethod(_operator2, op=operator.ifloordiv, inplace=True)  # type: Chunk[V]
    __imod__ = BetterPartialMethod(_operator2, op=operator.imod, inplace=True)  # type: Chunk[V]


class ChunkGrid(Generic[V]):
    def __init__(self, chunk_size: int = 8, dtype=None, fill_value: Optional[V] = None):
        assert chunk_size > 0
        self._chunk_size = chunk_size
        self._dtype = np.dtype(dtype).type
        self._fill_value = self._dtype() if fill_value is None else self._dtype(fill_value)
        self.chunks: IndexDict[Chunk[V]] = IndexDict()

    @property
    def dtype(self):
        return self._dtype

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        s = self._chunk_size
        return s, s, s

    @property
    def fill_value(self) -> V:
        return self._fill_value

    def size(self):
        index_min, index_max = self.chunks.minmax()
        return (index_max - index_min + 1) * self._chunk_size

    def astype(self, dtype: Type[M]) -> "ChunkGrid[M]":
        if self._dtype == dtype:
            return self
        grid_new: ChunkGrid[M] = ChunkGrid(self._chunk_size, dtype, fill_value=dtype(self._fill_value))
        for src in self.chunks.values():
            grid_new.chunks.insert(src.index, src.astype(dtype))
        return grid_new

    def convert(self, func: Callable[[V], M]) -> "ChunkGrid[M]":
        func_vec = np.vectorize(func)
        grid_new: ChunkGrid[M] = ChunkGrid(self._chunk_size, fill_value=func(self._fill_value))
        for src in self.chunks:
            grid_new.chunks.insert(src.index, src.convert(func, func_vec))
        return grid_new

    def copy(self, empty=False, dtype: Optional[Type[M]] = None) -> "ChunkGrid[M]":
        dtype = dtype or self._dtype
        new = ChunkGrid(self._chunk_size, dtype, dtype(self._fill_value))
        if not empty:
            for src in self.chunks.values():
                new.chunks.insert(src.index, src.copy())
        return new

    def split(self, splits: int, chunk_size: Optional[int] = None) -> "ChunkGrid[V]":
        assert splits > 0 and self._chunk_size % splits == 0
        chunk_size = chunk_size or self._chunk_size
        grid_new: ChunkGrid[V] = ChunkGrid(chunk_size, self._dtype, self._fill_value)
        for c in self.chunks.values():
            for c_new in c.split(splits, chunk_size):
                grid_new.chunks.insert(c_new.index, c_new)
        return grid_new

    def _new_chunk_factory(self, index: Index):
        return Chunk(index, self._chunk_size, self._dtype, self._fill_value)

    def chunk_index(self, pos: Vec3i):
        return np.asarray(pos, dtype=np.int) // self._chunk_size

    def chunk_at_pos(self, pos: Vec3i) -> Optional[Chunk[V]]:
        return self.chunks.get(self.chunk_index(pos))

    def ensure_chunk_at_index(self, index: ChunkIndex, *, insert=True) -> Chunk[V]:
        return self.chunks.create_if_absent(index, self._new_chunk_factory, insert=insert)

    def ensure_chunk_at_pos(self, pos: Vec3i, insert=True) -> Chunk[V]:
        return self.ensure_chunk_at_index(self.chunk_index(pos), insert=insert)

    def empty_mask(self, default=False) -> np.ndarray:
        return np.full(self.chunk_shape, default, dtype=np.bool)

    @classmethod
    def iter_neighbors_indicies(cls, index: ChunkIndex) -> Iterator[Tuple[ChunkFace, Vec3i]]:
        yield from ((f, np.add(index, f.direction)) for f in ChunkFace)

    def iter_neighbors(self, index: ChunkIndex, flatten=False) -> Iterator[Tuple[ChunkFace, Optional["Chunk"]]]:
        if flatten:
            yield from ((f, c) for f, c in self.iter_neighbors(index, False) if c is not None)
        else:
            yield from ((f, self.chunks.get(i, None)) for f, i in self.iter_neighbors_indicies(index))

    def __bool__(self):
        raise RuntimeWarning(f"Do not use __bool__ on {self.__class__}")

    def all(self):
        return all(c.all() for c in self.chunks.values())

    def any(self):
        return any(c.any() for c in self.chunks.values())

    def to_sparse(self, x: Union[int, slice, None] = None, y: Union[int, slice, None] = None,
                  z: Union[int, slice, None] = None) -> Tuple[sparse.SparseArray, Vec3i]:
        if len(self.chunks) == 0:
            return sparse.zeros(0), np.zeros(3)

        # Variable cache
        cs = self._chunk_size

        index_min, index_max = self.chunks.minmax()
        pos_min = index_min * cs
        pos_max = (index_max + 1) * cs
        it = PositionIter(x, y, z, low=pos_min, high=pos_max)

        chunk_min = it.start // cs
        chunk_max = it.stop // cs
        chunk_len = chunk_max - chunk_min

        arr = sparse.DOK(tuple(chunk_len * cs), fill_value=self._fill_value, dtype=self.dtype)
        for c in self.chunks.values():
            if np.all(chunk_min <= c.index) and np.all(c.index <= chunk_max):
                u, v, w = (c.index - chunk_min) * cs
                arr[u:u + cs, v:v + cs, w:w + cs] = c.to_array()

        start = it.start - chunk_min * cs
        stop = it.stop - chunk_min * cs
        step = it.step
        return (
            arr.to_coo()[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1], start[2]:stop[2]:step[2]],
            chunk_min * cs
        )

    def to_dense(self, *args, **kwargs) -> np.ndarray:
        res, offset = self.to_sparse(*args, **kwargs)
        return res.todense()

    def where(self, other: "ChunkGrid[bool]") -> "ChunkGrid[V]":
        result = self.copy(empty=True)
        for i, o in other.chunks.items():
            c = self.chunks.get(i, None)
            if c is not None and c.any():
                result.chunks.insert(i, c.where(o))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.to_sparse(item)
        elif isinstance(item, tuple) and len(item) <= 3:
            return self.to_sparse(*item)
        elif isinstance(item, ChunkGrid):
            return self.where(item)
        else:
            raise IndexError("Invalid get")

    def get_pos(self, pos: Vec3i) -> V:
        index = self.chunk_index(pos)
        c: Chunk[V] = self.chunks.get(index, None)
        if c is None:
            return self._fill_value
        else:
            return c.get_pos(pos)

    def set_pos(self, pos: Vec3i, value: V) -> Chunk[V]:
        c = self.ensure_chunk_at_pos(pos)
        c.set_pos(pos, value)
        return c

    def set_or_fill(self, pos: Vec3i, value: V) -> Chunk[V]:
        c = self.ensure_chunk_at_pos(pos)
        c.set_or_fill(pos, value)
        return c

    def _set_slices(self, value: Union[V, sparse.SparseArray, np.ndarray],
                    x: Union[int, slice, None] = None,
                    y: Union[int, slice, None] = None,
                    z: Union[int, slice, None] = None):
        # Variable cache
        cs = self._chunk_size

        it = PositionIter.require_bounded(x, y, z)

        if isinstance(value, (sparse.SparseArray, np.ndarray)):
            assert value.shape == it.shape
            if self._dtype is not None:
                value = value.astype(self._dtype)
            for i, pos in it.iter_with_indices():
                self.set_pos(pos, value[i])
        else:
            for pos in it:
                self.set_pos(pos, value)

    def _set_positions(self, pos: np.ndarray, value: Union[V, Sequence]):
        pos = np.asarray(pos, dtype=int)
        if pos.shape == (3,):
            self.set_pos(pos, value)
        assert pos.ndim == 2 and pos.shape[1] == 3
        if isinstance(value, (list, tuple, np.ndarray)):
            assert len(pos) == len(value)
            for p, v in zip(pos, value):
                self.set_pos(p, v)
        else:
            for p in pos:
                self.set_pos(p, value)

    def _set_chunks(self, other: "ChunkGrid", value: Union[V, np.ndarray, Chunk[V]]):
        assert self._chunk_size == other._chunk_size
        for o in other.chunks:
            self.ensure_chunk_at_index(o.index)[o] = value

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self._set_slices(value, key)
        elif isinstance(key, tuple) and len(key) <= 3:
            self._set_slices(value, *key)
        elif isinstance(key, (np.ndarray, list)):
            self._set_positions(key, value)
        elif isinstance(key, ChunkGrid):
            self._set_chunks(key, value)
        else:
            raise IndexError("Invalid get")

    # def equals(self, other: Union[Chunk, V]) -> "ChunkGrid[bool]":
    #     if isinstance(other, ChunkGrid):
    #         assert self._chunk_size == other._chunk_size
    #         new_grid: ChunkGrid[bool] = ChunkGrid(self._chunk_size, dtype=bool, empty_value=False)
    #         for i, a in self.chunks.items():
    #             b = other.chunks.get(i, None)
    #             if b is not None:
    #                 new_grid.chunks.insert(i, a.equals(b))
    #     else:
    #         new_grid: ChunkGrid[bool] = ChunkGrid(self._chunk_size, dtype=bool, empty_value=False)
    #         for i, a in self.chunks.items():
    #             new_grid.chunks.insert(i, a.equals(other))
    #     return new_grid
    #
    # def __eq__(self, other) -> "ChunkGrid[bool]":
    #     return self.equals(other)

    def equals(self, other: Union[Chunk, V]) -> "ChunkGrid[bool]":
        return self.eq(other)

    def __eq__(self, other) -> "ChunkGrid[bool]":
        return self.eq(other)

    def invert(self, inplace=False) -> "ChunkGrid[V]":
        """Applies the NOT operator on the chunks"""
        if inplace:
            for c in self.chunks.values():
                c.invert(inplace=True)
            return self
        else:
            new_grid: ChunkGrid[V] = ChunkGrid(self._chunk_size, dtype=self._dtype, fill_value=False)
            for i, c in self.chunks.items():
                new_grid.chunks.insert(i, c.invert())
            return new_grid

    def _operator1(self,
                   op: Callable[[Chunk[V]], Union[Chunk[M]]],
                   dtype: Optional[Type[M]] = None, inplace=False, ) -> "ChunkGrid[M]":
        dtype = dtype or self.dtype
        # Inplace selection
        new_grid = self if inplace else self.copy(empty=True, dtype=dtype)

        for i, a in self.chunks.items():
            new_chunk = op(a.copy())
            assert isinstance(new_chunk, Chunk)
            new_grid.chunks.insert(i, new_chunk)

        if dtype is not None:
            new_grid._dtype = dtype
            new_grid._fill_value = dtype(self._fill_value)
        return new_grid

    def _operator2(self, other: Union["ChunkGrid[V]", np.ndarray, V],
                   op: Callable[[Chunk[V], Union[Chunk[M], np.ndarray, M]], Union[Chunk[M]]],
                   dtype: Optional[Type[M]] = None, inplace=False) -> "ChunkGrid[M]":
        dtype = dtype or self.dtype
        # Inplace selection
        new_grid = self if inplace else self.copy(empty=True, dtype=dtype)

        if inplace:
            # Update data type
            for c in new_grid.chunks.values():
                c._dtype = dtype

        if isinstance(other, ChunkGrid):
            assert new_grid._chunk_size == other._chunk_size
            indices = set(self.chunks.keys())
            indices.update(other.chunks.keys())
            for i in indices:
                a = self.ensure_chunk_at_index(i, insert=False)
                b = other.ensure_chunk_at_index(i, insert=False)
                new_chunk = op(a.copy(), b)
                assert isinstance(new_chunk, Chunk)
                new_grid.chunks.insert(i, new_chunk)
        else:
            for i, a in self.chunks.items():
                new_chunk = op(a.copy(), other)
                assert isinstance(new_chunk, Chunk)
                new_grid.chunks.insert(i, new_chunk)

        # Update dtype and if possible the default fill value
        # noinspection DuplicatedCode
        if dtype is not None:
            new_grid._dtype = dtype
            try:
                if isinstance(other, ChunkGrid):
                    new_grid._fill_value = dtype(op(self._fill_value, other._fill_value))
                else:
                    new_grid._fill_value = dtype(op(self._fill_value, other))
            except Exception:
                try:
                    new_grid._fill_value = dtype(self._fill_value)
                except Exception:
                    pass
        return new_grid

    # Operators
    __eq__ = BetterPartialMethod(_operator2, op=operator.eq, dtype=bool)  # type: ChunkGrid[bool]
    __ne__ = BetterPartialMethod(_operator2, op=operator.ne, dtype=bool)  # type: ChunkGrid[bool]
    __lt__ = BetterPartialMethod(_operator2, op=operator.lt, dtype=bool)  # type: ChunkGrid[bool]
    __le__ = BetterPartialMethod(_operator2, op=operator.le, dtype=bool)  # type: ChunkGrid[bool]
    __gt__ = BetterPartialMethod(_operator2, op=operator.gt, dtype=bool)  # type: ChunkGrid[bool]
    __ge__ = BetterPartialMethod(_operator2, op=operator.ge, dtype=bool)  # type: ChunkGrid[bool]

    __abs__ = BetterPartialMethod(_operator1, op=operator.abs)  # type: ChunkGrid[V]
    __invert__ = BetterPartialMethod(_operator1, op=operator.inv)  # type: ChunkGrid[V]
    __neg__ = BetterPartialMethod(_operator1, op=operator.neg)  # type: ChunkGrid[V]

    __add__ = BetterPartialMethod(_operator2, op=operator.add)  # type: ChunkGrid[V]
    __and__ = BetterPartialMethod(_operator2, op=operator.and_)  # type: ChunkGrid[V]
    __floordiv__ = BetterPartialMethod(_operator2, op=operator.floordiv)  # type: ChunkGrid[V]
    __mod__ = BetterPartialMethod(_operator2, op=operator.mod)  # type: ChunkGrid[V]
    __mul__ = BetterPartialMethod(_operator2, op=operator.mul)  # type: ChunkGrid[V]
    __matmul__ = BetterPartialMethod(_operator2, op=operator.matmul)  # type: ChunkGrid[V]
    __or__ = BetterPartialMethod(_operator2, op=operator.or_)  # type: ChunkGrid[V]
    __sub__ = BetterPartialMethod(_operator2, op=operator.sub)  # type: ChunkGrid[V]
    __truediv__ = BetterPartialMethod(_operator2, op=operator.truediv, dtype=np.float)  # type: ChunkGrid[float]
    __xor__ = BetterPartialMethod(_operator2, op=operator.xor)  # type: ChunkGrid[V]

    __iand__ = BetterPartialMethod(_operator2, op=operator.iand, inplace=True)  # type: ChunkGrid[V]
    __ior__ = BetterPartialMethod(_operator2, op=operator.ior, inplace=True)  # type: ChunkGrid[V]
    __ixor__ = BetterPartialMethod(_operator2, op=operator.ixor, inplace=True)  # type: ChunkGrid[V]
    __iadd__ = BetterPartialMethod(_operator2, op=operator.iadd, inplace=True)  # type: ChunkGrid[V]
    __isub__ = BetterPartialMethod(_operator2, op=operator.isub, inplace=True)  # type: ChunkGrid[V]
    __imul__ = BetterPartialMethod(_operator2, op=operator.imul, inplace=True)  # type: ChunkGrid[V]
    __itruediv__ = BetterPartialMethod(_operator2, op=operator.itruediv, dtype=np.float,
                                       inplace=True)  # type: ChunkGrid[float]
    __ifloordiv__ = BetterPartialMethod(_operator2, op=operator.ifloordiv, inplace=True)  # type: ChunkGrid[V]
    __imod__ = BetterPartialMethod(_operator2, op=operator.imod, inplace=True)  # type: ChunkGrid[V]

    def pad_chunks(self, width: int = 1):
        visited: Set[ChunkIndex] = set()
        for s in range(0, width):
            extra: Set[ChunkIndex] = set(tuple(n) for i in self.chunks.keys()
                                         for f, n in self.iter_neighbors_indicies(i))
            extra = extra.difference(visited)
            for e in extra:
                self.ensure_chunk_at_index(e)
            visited.update(extra)

    def hull(self) -> Iterator[Chunk[V]]:
        """Some of the outer chunks that represent a hull around all chunks"""
        if self.chunks:
            it = self.chunks.sliced_iterator()
            for x in it.x.range():
                for y in it.y.range():
                    for z in it.z.range():
                        c = self.chunks.get((x, y, z), None)
                        if c is not None:
                            yield c
                            break
            for x in reversed(it.x.range()):
                for y in reversed(it.y.range()):
                    for z in reversed(it.z.range()):
                        c = self.chunks.get((x, y, z), None)
                        if c is not None:
                            yield c
                            break
