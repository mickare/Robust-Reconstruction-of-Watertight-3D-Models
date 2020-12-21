import enum
import functools
import operator
from typing import Union, Tuple, Iterator, Optional, Generic, TypeVar, Callable, Type, Sequence

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

    def slice(self) -> Tuple[Union[int, slice], ...]:
        s = slice(None)
        return ((-1, s, s),
                (0, s, s),
                (s, -1, s),
                (s, 0, s),
                (s, s, -1),
                (s, s, 0))[self]

    def shape(self, size: int) -> Tuple[int, int, int]:
        return ((1, size, size),
                (1, size, size),
                (size, 1, size),
                (size, 1, size),
                (size, size, 1),
                (size, size, 1))[self]

    def __bool__(self):
        return True


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
    def __init__(self, index: Vec3i, size: int, dtype: Optional[Type[V]] = None, empty_value: Optional[V] = None):
        self._index: Vec3i = np.asarray(index, dtype=np.int)
        self._size = size
        self._dtype = np.dtype(dtype).type
        self._empty_value = self._dtype() if empty_value is None else self._dtype(empty_value)
        self._is_filled = True
        self._value: Union[V, np.ndarray] = self._empty_value

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

    def is_filled(self) -> bool:
        return self._is_filled

    def is_array(self) -> bool:
        return not self._is_filled

    def inner(self, pos: Vec3i) -> np.ndarray:
        return np.asarray(pos, dtype=np.int) % self._size

    def set_pos(self, pos: Vec3i, value: V):
        inner = self.inner(pos)
        arr = self.to_array()
        arr[tuple(inner)] = value
        self.set_array(arr)

    def set_fill(self, value: V):
        self._value = value
        self._is_filled = True
        return self

    def set_array(self, value: np.ndarray):
        assert value.shape == self.shape
        self._value = np.asarray(value, dtype=self._dtype)
        self._is_filled = False
        return self

    def to_array(self):
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
            arr = np.full(self.shape, self._empty_value, dtype=self._dtype)
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

    def copy(self, empty=False):
        c = Chunk(self._index, self._size, dtype=self._dtype, empty_value=self._empty_value)
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
            c = Chunk(new_index, size=chunk_size, empty_value=self._empty_value)
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
        c = Chunk(self._index, self._size, empty_value=func(self._empty_value))
        if self.is_filled():
            c.set_fill(func(self._value))
        else:
            c.set_array(func_vec(self._value))
        return c

    def astype(self, dtype: Type[M]) -> "Chunk[M]":
        if self._dtype == dtype:
            return self
        c = Chunk(self._index, self._size, dtype=dtype, empty_value=dtype(self._empty_value))
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

    def invert(self, inplace=False) -> "Chunk[V]":
        c = self if inplace else self.copy(empty=True)
        c._value = np.invert(self._value)
        return c

    def equals(self, other: Union["Chunk", V]) -> "Chunk[bool]":
        # return self._op(other, operator.eq, bool)
        return self.eq(other)

    def _operator(self, other: Union["Chunk[V]", np.ndarray, V],
                  op: Callable[[Union[np.ndarray, V], Union[np.ndarray, V]], Union[np.ndarray, M]],
                  dtype: Optional[Type[M]] = None, inplace=False, ) -> "Chunk[M]":
        # Inplace selection
        if inplace:
            c = self
        else:
            c = Chunk(self.index, size=self.size, dtype=dtype or self.dtype)

        if isinstance(other, Chunk):
            c._is_filled = self._is_filled and other._is_filled
            c._value = op(self._value, other._value)
        else:
            c._is_filled = self._is_filled
            c._value = op(self._value, other)
        c.cleanup_memory()
        return c

    # Operators
    eq = BetterPartialMethod(_operator, op=operator.eq, dtype=bool)
    ne = BetterPartialMethod(_operator, op=operator.ne, dtype=bool)

    abs = BetterPartialMethod(_operator, op=operator.abs)
    add = BetterPartialMethod(_operator, op=operator.add)
    and_ = BetterPartialMethod(_operator, op=operator.and_)
    floordiv = BetterPartialMethod(_operator, op=operator.floordiv)
    inv = BetterPartialMethod(_operator, op=operator.inv)
    mod = BetterPartialMethod(_operator, op=operator.mod)
    mul = BetterPartialMethod(_operator, op=operator.mul)
    matmul = BetterPartialMethod(_operator, op=operator.matmul)
    neg = BetterPartialMethod(_operator, op=operator.neg)
    or_ = BetterPartialMethod(_operator, op=operator.or_)
    pos = BetterPartialMethod(_operator, op=operator.pos)
    sub = BetterPartialMethod(_operator, op=operator.sub)
    truediv = BetterPartialMethod(_operator, op=operator.truediv)
    xor = BetterPartialMethod(_operator, op=operator.xor)

    iand = BetterPartialMethod(_operator, op=operator.iand, dtype=bool, inplace=True)
    ior = BetterPartialMethod(_operator, op=operator.ior, dtype=bool, inplace=True)
    ixor = BetterPartialMethod(_operator, op=operator.ixor, dtype=bool, inplace=True)
    iadd = BetterPartialMethod(_operator, op=operator.iadd, dtype=bool, inplace=True)
    isub = BetterPartialMethod(_operator, op=operator.isub, dtype=bool, inplace=True)
    imul = BetterPartialMethod(_operator, op=operator.imul, dtype=bool, inplace=True)
    itruediv = BetterPartialMethod(_operator, op=operator.itruediv, dtype=bool, inplace=True)
    ifloordiv = BetterPartialMethod(_operator, op=operator.ifloordiv, dtype=bool, inplace=True)
    imod = BetterPartialMethod(_operator, op=operator.imod, dtype=bool, inplace=True)

    # Set alternative opertator methods names
    __eq__ = eq
    __ne__ = ne

    __abs__ = abs
    __add__ = add
    __and__ = and_
    __floordiv__ = floordiv
    __inv__ = inv
    __mul__ = mul
    __matmul__ = matmul
    __or__ = or_
    __pos__ = pos
    __sub__ = sub
    __truediv__ = truediv
    __xor__ = xor

    __iand__ = iand
    __ior__ = ior
    __ixor__ = ixor
    __iadd__ = iadd
    __isub__ = isub
    __imul__ = imul
    __itruediv__ = itruediv
    __ifloordiv__ = ifloordiv
    __imod__ = imod


class ChunkGrid(Generic[V]):
    def __init__(self, chunk_size: int = 8, dtype=None, empty_value: Optional[V] = None):
        assert chunk_size > 0
        self._chunk_size = chunk_size
        self._dtype = np.dtype(dtype).type
        self._empty_value = self._dtype() if empty_value is None else self._dtype(empty_value)
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
    def empty_value(self) -> V:
        return self._empty_value

    def size(self):
        index_min, index_max = self.chunks.minmax()
        return (index_max - index_min + 1) * self._chunk_size

    def astype(self, dtype: Type[M]) -> "ChunkGrid[M]":
        if self._dtype == dtype:
            return self
        grid_new: ChunkGrid[M] = ChunkGrid(self._chunk_size, dtype, empty_value=dtype(self._empty_value))
        for src in self.chunks.values():
            grid_new.chunks.insert(src.index, src.astype(dtype))
        return grid_new

    def convert(self, func: Callable[[V], M]) -> "ChunkGrid[M]":
        func_vec = np.vectorize(func)
        grid_new: ChunkGrid[M] = ChunkGrid(self._chunk_size, empty_value=func(self._empty_value))
        for src in self.chunks:
            grid_new.chunks.insert(src.index, src.convert(func, func_vec))
        return grid_new

    def copy(self, empty=False):
        new = ChunkGrid(self._chunk_size, self._dtype, self._empty_value)
        if not empty:
            for src in self.chunks.values():
                new.chunks.insert(src.index, src.copy())
        return new

    def split(self, splits: int, chunk_size: Optional[int] = None) -> "ChunkGrid[V]":
        assert splits > 0 and self._chunk_size % splits == 0
        chunk_size = chunk_size or self._chunk_size
        grid_new: ChunkGrid[V] = ChunkGrid(chunk_size, self._dtype, self._empty_value)
        for c in self.chunks.values():
            for c_new in c.split(splits, chunk_size):
                grid_new.chunks.insert(c_new.index, c_new)
        return grid_new

    def _new_chunk_factory(self, index: Index):
        return Chunk(index, self._chunk_size, self._dtype, self._empty_value)

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

    def iter_neighbors_indicies(self, index: ChunkIndex) -> Iterator[Tuple[ChunkFace, Vec3i]]:
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

        arr = sparse.DOK(tuple(chunk_len * cs), fill_value=self._empty_value, dtype=self.dtype)
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

    def set_pos(self, pos: Vec3i, value: V):
        self.ensure_chunk_at_pos(pos).set_pos(pos, value)

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
            new_grid: ChunkGrid[V] = ChunkGrid(self._chunk_size, dtype=self._dtype, empty_value=False)
            for i, c in self.chunks.items():
                new_grid.chunks.insert(i, c.invert())
            return new_grid

    def _operator(self, other: Union["ChunkGrid[V]", np.ndarray, V],
                  op: Callable[[Chunk[V], Union[Chunk[M], np.ndarray, M]], Union[Chunk[M]]],
                  dtype: Optional[Type[M]] = None, inplace=False, ) -> "ChunkGrid[M]":
        # Inplace selection
        if inplace:
            new_grid = self
        else:
            new_grid = self.copy(True).astype(dtype or self.dtype)

        if isinstance(other, ChunkGrid):
            assert new_grid._chunk_size == other._chunk_size
            for i, a in self.chunks.items():
                b = other.ensure_chunk_at_index(i, insert=False)
                new_chunk = op(a, b)
                assert isinstance(new_chunk, Chunk)
                new_grid.chunks.insert(i, new_chunk)
        else:
            for i, a in self.chunks.items():
                new_chunk = op(a, other)
                assert isinstance(new_chunk, Chunk)
                new_grid.chunks.insert(i, op(a, other))

        return new_grid

    eq = BetterPartialMethod(_operator, op=Chunk.eq, dtype=bool)
    ne = BetterPartialMethod(_operator, op=Chunk.ne, dtype=bool)

    abs = BetterPartialMethod(_operator, op=Chunk.abs)
    add = BetterPartialMethod(_operator, op=Chunk.add)
    and_ = BetterPartialMethod(_operator, op=Chunk.and_)
    floordiv = BetterPartialMethod(_operator, op=Chunk.floordiv)
    inv = BetterPartialMethod(_operator, op=Chunk.inv)
    mod = BetterPartialMethod(_operator, op=Chunk.mod)
    mul = BetterPartialMethod(_operator, op=Chunk.mul)
    matmul = BetterPartialMethod(_operator, op=Chunk.matmul)
    neg = BetterPartialMethod(_operator, op=Chunk.neg)
    or_ = BetterPartialMethod(_operator, op=Chunk.or_)
    pos = BetterPartialMethod(_operator, op=Chunk.pos)
    sub = BetterPartialMethod(_operator, op=Chunk.sub)
    truediv = BetterPartialMethod(_operator, op=Chunk.truediv)
    xor = BetterPartialMethod(_operator, op=Chunk.xor)

    iand = BetterPartialMethod(_operator, op=Chunk.iand, dtype=bool, inplace=True)
    ior = BetterPartialMethod(_operator, op=Chunk.ior, dtype=bool, inplace=True)
    ixor = BetterPartialMethod(_operator, op=Chunk.ixor, dtype=bool, inplace=True)
    iadd = BetterPartialMethod(_operator, op=Chunk.iadd, dtype=bool, inplace=True)
    isub = BetterPartialMethod(_operator, op=Chunk.isub, dtype=bool, inplace=True)
    imul = BetterPartialMethod(_operator, op=Chunk.imul, dtype=bool, inplace=True)
    itruediv = BetterPartialMethod(_operator, op=Chunk.itruediv, dtype=bool, inplace=True)
    ifloordiv = BetterPartialMethod(_operator, op=Chunk.ifloordiv, dtype=bool, inplace=True)
    imod = BetterPartialMethod(_operator, op=Chunk.imod, dtype=bool, inplace=True)

    # Set alternative opertator methods names
    __eq__ = eq
    __ne__ = ne

    __abs__ = abs
    __add__ = add
    __and__ = and_
    __floordiv__ = floordiv
    __inv__ = inv
    __mul__ = mul
    __matmul__ = matmul
    __or__ = or_
    __pos__ = pos
    __sub__ = sub
    __truediv__ = truediv
    __xor__ = xor

    __iand__ = iand
    __ior__ = ior
    __ixor__ = ixor
    __iadd__ = iadd
    __isub__ = isub
    __imul__ = imul
    __itruediv__ = itruediv
    __ifloordiv__ = ifloordiv
    __imod__ = imod
