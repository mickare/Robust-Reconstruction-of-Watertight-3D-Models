import enum
import itertools
import operator
from typing import Union, Tuple, Iterator, Optional, Generic, TypeVar, Callable, Type, Sequence, Set

import numpy as np
from scipy import ndimage

from data.data_utils import PositionIter
from data.faces import ChunkFace
from data.index_dict import IndexDict, Index
from mathlib import Vec3i

V = TypeVar('V')
M = TypeVar('M')
ChunkIndex = Index


class ChunkHelper:
    class _IndexMeshGrid:
        def __getitem__(self, item) -> Iterator[Vec3i]:
            assert isinstance(item, slice)
            xs, ys, zs = np.mgrid[item, item, item]
            return zip(xs.flatten(), ys.flatten(), zs.flatten())

    indexGrid = _IndexMeshGrid()


class Chunk(Generic[V]):
    __slots__ = ('_index', '_size', '_dtype', '_fill_value', '_is_filled', '_value')

    def __init__(self, index: Vec3i, size: int, dtype: Optional[Union[np.dtype, Type[V]]] = None,
                 fill_value: Optional[V] = None):
        self._index: Vec3i = np.asarray(index, dtype=np.int)
        self._size = size
        self._dtype = np.dtype(dtype)
        self._fill_value = fill_value
        self.set_fill(self._fill_value)
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
    def dtype(self) -> np.dtype:
        return self._dtype

    def __dtype(self, other: Optional[Union[np.dtype, Type[M]]] = None) -> np.dtype:
        return np.dtype(other) if other is not None else self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        cs = self._size
        return cs, cs, cs

    @property
    def array_shape(self) -> Tuple[int, ...]:
        cs = self._size
        return cs, cs, cs, *self._dtype.shape

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
        if not self._dtype.subdtype:
            value = self._dtype.type(value)
        self._value = value
        self._is_filled = True
        return self

    def set_array(self, value: np.ndarray) -> "Chunk[V]":
        if isinstance(value, np.ndarray):
            value = value.astype(self._dtype.base)
        else:
            value = np.asarray(value, dtype=self._dtype.base)
        assert self.array_shape == value.shape, f"{self.array_shape} != {value.shape}"
        self._value = value
        self._is_filled = False
        return self

    def to_array(self) -> np.ndarray:
        if self._is_filled:
            return np.full(self.shape, self._value, dtype=self._dtype)
        else:
            return self._value

    def filter(self, other: "Chunk[bool]", fill_value: Optional[V] = None) -> "Chunk[V]":
        other: Chunk[bool] = other.astype(bool)
        c = self.copy(empty=True, fill_value=fill_value)
        if other.is_filled() and other._value:
            c.set_fill(self._value)
        else:
            arr = np.full(self.shape, c._fill_value, dtype=self._dtype)
            arr[other._value] = self._value[other._value]
            c.set_array(arr)
        return c

    def items(self, mask: Optional["Chunk[bool]"] = None) -> Iterator[Tuple[Vec3i, V]]:
        it = PositionIter(None, None, None, np.zeros(3), self.shape)
        if mask is None:
            ps = np.asarray(list(it))
        elif isinstance(mask, Chunk):
            m = mask.to_array()
            ps = np.array([p for p in it if m[p]])
        else:
            raise ValueError(f"invalid mask of type {type(mask)}")
        if len(ps) > 0:
            cps = ps + self.position_low
            if self.is_filled():
                yield from ((p, self._value) for p in cps)
            else:
                yield from zip(cps, self.to_array()[tuple(ps.T)])

    def where(self, mask: Optional["Chunk[bool]"] = None) -> np.ndarray:
        it = PositionIter(None, None, None, np.zeros(3), self.shape)
        if mask is None:
            if self.is_filled():
                if self._value:
                    return np.asarray(list(it)) + self.position_low
            else:
                return np.argwhere(self.to_array().astype(bool)) + self.position_low
        elif isinstance(mask, Chunk):
            if mask.is_filled() and bool(mask.value):
                return self.where(mask=None)
            else:
                m = mask.to_array()
                return np.array([p for p in it if m[p]]) + self.position_low
        else:
            raise ValueError(f"invalid mask of type {type(mask)}")
        return np.empty((0, 3), dtype=int)

    def __getitem__(self, item):
        if isinstance(item, Chunk):
            return self.filter(item)
        return self.to_array()[item]

    def __setitem__(self, key: Union[np.ndarray, "Chunk"], value: Union[V, np.ndarray, "Chunk"]):
        is_value_chunk = isinstance(value, Chunk)
        if is_value_chunk:
            assert value._size == self._size

        if isinstance(key, Chunk):
            if key.all():
                # Fast set
                if is_value_chunk:
                    self._is_filled = value._is_filled
                    self._value = value._value
                    return
            key = key.to_array().astype(np.bool8)

        arr = self.to_array()
        if is_value_chunk:  # Masked
            arr[key] = value.to_array()[key]
        else:
            arr[key] = value
        self.set_array(arr)

    def copy(self, empty=False, dtype=None, fill_value: Optional[V] = None):
        dtype = self.__dtype(dtype)
        fill_value = self._fill_value if fill_value is None else fill_value
        c = Chunk(self._index, self._size, dtype=dtype, fill_value=fill_value)
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

        dtype = self._dtype

        # Voxel repeats to achieve chunk size
        repeats = chunk_size / split_size
        assert repeats > 0 and repeats % 1 == 0  # Chunk size must be achieved by duplication of voxels
        repeats = int(repeats)

        for offset in ChunkHelper.indexGrid[:splits]:
            new_index = np.add(self._index * splits, offset)
            c = Chunk(new_index, size=chunk_size, dtype=dtype, fill_value=self._fill_value)
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
        raise ValueError(f"The truth value of {__class__} is ambiguous. "
                         "Use a.any(), or a.all(), or wrap the comparison (0 < a) & (a < 0)")

    def cleanup(self):
        """Try to reduce memory footprint"""
        if self.is_array():
            u = np.unique(self._value)
            if len(u) == 1:
                self.set_fill(u.item())
        return self

    def all(self) -> bool:
        if self.is_filled():
            return bool(self._value)
        return np.all(self._value)

    def any(self) -> bool:
        if self.is_filled():
            return bool(self._value)
        return np.any(self._value)

    def any_fast(self) -> bool:
        if self.is_filled():
            return bool(self._value)
        return True

    def padding(self, grid: "ChunkGrid[V]", padding: int, corners=False, value: Optional[V] = None) -> np.ndarray:
        assert 0 <= padding <= self._size
        if padding == 0:
            return self.to_array()
        if value is None:
            value = self._fill_value

        arr = self.to_array()
        pad = padding
        if arr.ndim > 3:
            pad = (pad, pad), (pad, pad), (pad, pad), *([(0, 0)] * (arr.ndim - 3))
        arr = np.pad(self.to_array(), pad, constant_values=value)

        for face, index in grid.iter_neighbors_indicies(self._index):
            c = grid.ensure_chunk_at_index(index, insert=False)
            pad = c.to_array()[face.flip().slice(padding)]
            arr[face.slice(padding, other=slice(padding, -padding))] = pad
        if corners:
            for u, v, w in ChunkFace.corners():
                s0 = ChunkFace.corner_slice(u, v, w, width=padding)
                s1 = ChunkFace.corner_slice(u.flip(), v.flip(), w.flip(), width=padding)
                d = ChunkFace.corner_direction(u, v, w)
                arr[s0] = grid.ensure_chunk_at_index(d + self._index, insert=False).to_array()[s1]
        return arr

    def apply(self, func: Callable[[Union[np.ndarray, V]], Union[np.ndarray, M]],
              dtype: Optional[Type[M]] = None, inplace=False, ) -> "Chunk[M]":
        dtype = self.__dtype(dtype)
        # Inplace selection
        c = self if inplace else self.copy()
        res = func(self._value)
        if not c._is_filled:
            assert isinstance(res, np.ndarray)
            if dtype is not None:
                res = res.astype(dtype)
        elif dtype is not None:
            if not isinstance(res, np.ndarray):
                res = dtype.type(res)
        c._value = res

        if dtype is not None:
            c._dtype = dtype
            c._fill_value = dtype.type(self._fill_value)
        return c

    def join(self, rhs, func: Callable[[Union[np.ndarray, V], Union[np.ndarray, V]], Union[np.ndarray, M]],
             dtype: Optional[Type[M]] = None, inplace=False, ) -> "Chunk[M]":
        dtype = self.__dtype(dtype)
        # Inplace selection
        c = self if inplace else self.copy()

        if isinstance(rhs, Chunk):
            c._is_filled = c._is_filled & rhs._is_filled
            c._value = func(c._value, rhs._value)
        else:
            c._is_filled = c._is_filled
            c._value = func(c._value, rhs)

        if dtype is not None:
            c._dtype = dtype
            try:
                if isinstance(rhs, Chunk):
                    c._fill_value = func(self._fill_value, rhs._fill_value)
                else:
                    c._fill_value = func(self._fill_value, rhs)
            except Exception:
                try:
                    c._fill_value = dtype.type(self._fill_value)
                except Exception:
                    pass

        return c

    # Comparison Operator

    def __eq__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.eq, dtype=np.bool8).cleanup()

    def __ne__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.ne, dtype=np.bool8).cleanup()

    def __lt__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.lt, dtype=np.bool8).cleanup()

    def __le__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.le, dtype=np.bool8).cleanup()

    def __gt__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.gt, dtype=np.bool8).cleanup()

    def __ge__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.ge, dtype=np.bool8).cleanup()

    eq = __eq__
    ne = __ne__
    lt = __lt__
    le = __le__
    gt = __gt__
    ge = __ge__

    equals = __eq__

    # Single Operator

    def __abs__(self) -> "Chunk[V]":
        return self.apply(func=operator.abs)

    def __invert__(self) -> "Chunk[V]":
        return self.apply(func=operator.inv)

    def __neg__(self) -> "Chunk[V]":
        return self.apply(func=operator.neg)

    abs = __abs__
    invert = __invert__
    neg = __neg__

    # Logic Operator

    def __and__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.and_)

    def __or__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.or_)

    def __xor__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.xor)

    def __iand__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.iand, inplace=True)

    def __ior__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.ior, inplace=True)

    def __ixor__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.ixor, inplace=True)

    # Math Operator

    def __add__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.add)

    def __sub__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.sub)

    def __mul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.mul)

    def __matmul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.matmul)

    def __mod__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.mod)

    def __pow__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.pow)

    def __floordiv__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.floordiv)

    def __iadd__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.iadd, inplace=True)

    def __isub__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.isub, inplace=True)

    def __imul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.imul, inplace=True)

    def __imatmul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.imatmul, inplace=True)

    def __imod__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.imod, inplace=True)

    def __ifloordiv__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.ifloordiv, inplace=True)

    # TrueDiv Operator

    def __truediv__(self, rhs) -> "Chunk[float]":
        return self.join(rhs, func=operator.truediv, dtype=np.float)

    def __itruediv__(self, rhs) -> "Chunk[float]":
        return self.join(rhs, func=operator.itruediv, dtype=np.float, inplace=True)

    # Reflected Operators
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    __rtruediv__ = __truediv__
    __rfloordiv__ = __floordiv__
    __rmod__ = __mod__
    __rpow__ = __pow__
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    def sum(self, dtype: Optional[Type[M]] = None) -> M:
        if self._is_filled:
            val = self._value * self._size ** 3
            return val if dtype is None else dtype(val)
        else:
            return np.sum(self._value, dtype=dtype)

    def _correlate_filter(self, func, kernel: np.ndarray, grid: "ChunkGrid[V]",
                          fill_value: Optional[M] = None, **kwargs) -> "Chunk[M]":
        assert kernel.ndim >= 3 and kernel.shape[0] == kernel.shape[1] == kernel.shape[2]
        fill_value = fill_value if fill_value is not None else self._fill_value
        pad = np.max(kernel.shape) // 2
        padded = self.padding(grid, pad, corners=True, value=fill_value)
        tmp = func(padded, kernel, **kwargs)
        arr = tmp[pad:-pad, pad:-pad, pad:-pad]
        assert arr.shape == self.array_shape
        return Chunk(self._index, self._size, self._dtype, fill_value=fill_value).set_array(arr)

    @classmethod
    def _stack(cls, chunks: Sequence["Chunk[V]"], dtype: np.dtype, fill_value=None) -> "Chunk[np.ndarray]":
        assert dtype.shape
        index = chunks[0]._index
        size = chunks[0]._size
        new_chunk = Chunk(index, size, dtype, fill_value)
        if all(c._is_filled for c in chunks):
            new_chunk.set_fill(np.array([c.value for c in chunks], dtype=dtype.base))
        else:
            arr = np.array([c.to_array() for c in chunks], dtype=dtype.base).transpose((1, 2, 3, 0))
            new_chunk.set_array(arr)
        return new_chunk


class ChunkGrid(Generic[V]):
    __slots__ = ('_chunk_size', '_dtype', '_fill_value', 'chunks')

    def __init__(self, chunk_size: int, dtype: Optional[Union[np.dtype, Type[V]]] = None,
                 fill_value: Optional[V] = None):
        assert chunk_size > 0
        self._chunk_size = chunk_size
        self._dtype = np.dtype(dtype)
        self._fill_value = fill_value
        self.chunks: IndexDict[Chunk[V]] = IndexDict()

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __dtype(self, other: Optional[Union[np.dtype, Type[M]]] = None) -> np.dtype:
        return np.dtype(other) if other is not None else self._dtype

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

    def size(self) -> Vec3i:
        index_min, index_max = self.chunks.minmax()
        return (index_max - index_min + 1) * self._chunk_size

    def astype(self, dtype: Type[M]) -> "ChunkGrid[M]":
        if self._dtype == dtype:
            return self
        new_grid: ChunkGrid[M] = ChunkGrid(self._chunk_size, dtype, fill_value=dtype(self._fill_value))
        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert
        for src in self.chunks.values():
            __new_grid_chunks_insert(src.index, src.astype(dtype))
        return new_grid

    def convert(self, func: Callable[[V], M]) -> "ChunkGrid[M]":
        func_vec = np.vectorize(func)
        new_grid: ChunkGrid[M] = ChunkGrid(self._chunk_size, fill_value=func(self._fill_value))
        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert
        for src in self.chunks:
            __new_grid_chunks_insert(src.index, src.convert(func, func_vec))
        return new_grid

    def copy(self, empty=False, dtype: Optional[Union[np.dtype, Type[M]]] = None,
             fill_value: Optional[M] = None) -> "ChunkGrid[M]":
        dtype = self.__dtype(dtype)
        fill_value = self._fill_value if fill_value is None else fill_value
        new_grid = ChunkGrid(self._chunk_size, dtype, fill_value)
        if not empty:
            # Method cache (prevent lookup in loop)
            __new_grid_chunks_insert = new_grid.chunks.insert
            for src in self.chunks.values():
                __new_grid_chunks_insert(src.index, src.copy(dtype=dtype, fill_value=fill_value))
        return new_grid

    def split(self, splits: int, chunk_size: Optional[int] = None) -> "ChunkGrid[V]":
        assert splits > 0 and self._chunk_size % splits == 0
        chunk_size = chunk_size or self._chunk_size
        new_grid: ChunkGrid[V] = ChunkGrid(chunk_size, self._dtype, self._fill_value)

        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert

        for c in self.chunks.values():
            for c_new in c.split(splits, chunk_size):
                __new_grid_chunks_insert(c_new.index, c_new)
        return new_grid

    def _new_chunk_factory(self, index: Index) -> Chunk[V]:
        return Chunk(index, self._chunk_size, self._dtype, self._fill_value)

    def chunk_index(self, pos: Vec3i) -> Vec3i:
        res = np.asarray(pos, dtype=np.int) // self._chunk_size
        assert res.shape == (3,)
        return res

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
        raise ValueError(f"The truth value of {__class__} is ambiguous. "
                         "Use a.any(), or a.all(), or wrap the comparison (0 < a) & (a < 0)")

    def all(self):
        """True if all chunks contain only True values"""
        return all(c.all() for c in self.chunks.values())

    def any(self):
        """True if any chunk contains any True value"""
        return any(c.any() for c in self.chunks.values())

    def to_dense(self, x: Union[int, slice, None] = None, y: Union[int, slice, None] = None,
                 z: Union[int, slice, None] = None, return_offset=False) -> Union[np.ndarray, Tuple[np.ndarray, Vec3i]]:
        """Convert the grid to a dense numpy array"""
        if len(self.chunks) == 0:
            if return_offset:
                return np.empty((0, 0, 0)), np.zeros(3)
            else:
                return np.empty((0, 0, 0))

        # Variable cache
        cs = self._chunk_size

        index_min, index_max = self.chunks.minmax()
        pos_min = index_min * cs
        pos_max = (index_max + 1) * cs
        voxel_it = PositionIter(x, y, z, low=pos_min, high=pos_max)

        chunk_it = voxel_it // cs
        chunk_min = np.asarray(chunk_it.start)
        chunk_max = np.asarray(chunk_it.stop)
        chunk_len = chunk_max - chunk_min

        res = np.full(tuple(chunk_len * cs), self.fill_value, dtype=self.dtype)

        # Method cache (prevent lookup in loop)
        __self_chunks_get = self.chunks.get
        __chunk_to_array = Chunk.to_array

        for index in chunk_it:
            c = __self_chunks_get(index, None)
            if c is not None:
                u, v, w = (index - chunk_min) * cs
                res[u:u + cs, v:v + cs, w:w + cs] = __chunk_to_array(c)

        start = voxel_it.start - chunk_min * cs
        stop = voxel_it.stop - chunk_min * cs
        step = voxel_it.step
        if return_offset:
            return (
                res[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1], start[2]:stop[2]:step[2]],
                chunk_min * cs
            )
        else:
            return res[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1], start[2]:stop[2]:step[2]]

    def items(self, mask: Optional["ChunkGrid[bool]"] = None) -> Iterator[Tuple[Vec3i, V]]:
        # Method cache (prevent lookup in loop)
        __chunk_items = Chunk.items
        if mask is None:
            for i, c in self.chunks.items():
                yield from __chunk_items(c)
        else:
            # Method cache (prevent lookup in loop)
            __chunk_any_fast = Chunk.any_fast
            __mask_ensure_chunk_at_index = mask.ensure_chunk_at_index
            for i, c in self.chunks.items():
                m = __mask_ensure_chunk_at_index(i, insert=False)
                if __chunk_any_fast(m):
                    yield from __chunk_items(c, mask=m)

    def where(self, mask: Optional["ChunkGrid[bool]"] = None) -> Iterator[Vec3i]:
        # Method cache (prevent lookup in loop)
        __chunk_where = Chunk.where
        if mask is None:
            for i, c in self.chunks.items():
                yield from __chunk_where(c)
        else:
            # Method cache (prevent lookup in loop)
            __chunk_any_fast = Chunk.any_fast
            __mask_ensure_chunk_at_index = mask.ensure_chunk_at_index
            for i, c in self.chunks.items():
                m = __mask_ensure_chunk_at_index(i, insert=False)
                if __chunk_any_fast(m):
                    yield from __chunk_where(c, mask=m)

    def filter(self, other: "ChunkGrid[bool]", fill_value: Optional[V] = None) -> "ChunkGrid[V]":
        """Apply a filter mask to this grid and return the masked values"""
        result = self.copy(empty=True, fill_value=fill_value)

        # Method cache (prevent lookup in loop)
        __self_chunks_get = self.chunks.get
        __chunk_any = Chunk.any
        __result_chunks_insert = result.chunks.insert

        for i, o in other.chunks.items():
            c = __self_chunks_get(i, None)
            if c is not None and __chunk_any(c):
                __result_chunks_insert(i, c.filter(o, fill_value=fill_value))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.to_dense(item)
        elif isinstance(item, tuple) and len(item) <= 3:
            return self.to_dense(*item)
        elif isinstance(item, ChunkGrid):
            return self.filter(item)
        elif isinstance(item, np.ndarray):
            return self.get_values(item)
        else:
            raise IndexError("Invalid get")

    def get_values(self, pos: Union[Sequence[Vec3i], np.ndarray]) -> np.ndarray:
        """Returns a list of values at the positions"""
        # Method cache (prevent lookup in loop)
        __np_argwhere = np.argwhere
        __self_ensure_chunk_at_index = self.ensure_chunk_at_index
        __chunk_to_array = Chunk.to_array

        pos = np.asarray(pos, dtype=int)
        assert pos.ndim == 2 and pos.shape[1] == 3
        cind, cinv = np.unique(pos // self._chunk_size, axis=0, return_inverse=True)
        result = np.zeros(len(cinv), dtype=self._dtype)
        for n, i in enumerate(cind):
            pind = __np_argwhere(cinv == n).flatten()
            cpos = pos[pind]
            chunk = __self_ensure_chunk_at_index(i, insert=False)
            result[pind] = __chunk_to_array(chunk)[tuple(cpos.T)]
        return result

    def get_value(self, pos: Vec3i) -> V:
        index = self.chunk_index(pos)
        c: Chunk[V] = self.chunks.get(index, None)
        if c is None:
            return self._fill_value
        else:
            return c.get_pos(pos)

    def set_value(self, pos: Vec3i, value: V) -> Chunk[V]:
        c = self.ensure_chunk_at_pos(pos)
        c.set_pos(pos, value)
        return c

    def set_or_fill(self, pos: Vec3i, value: V) -> Chunk[V]:
        c = self.ensure_chunk_at_pos(pos)
        c.set_or_fill(pos, value)
        return c

    def _set_slices(self, value: Union[V, np.ndarray],
                    x: Union[int, slice, None] = None,
                    y: Union[int, slice, None] = None,
                    z: Union[int, slice, None] = None):
        # Variable cache
        cs = self._chunk_size

        it = PositionIter.require_bounded(x, y, z)

        if isinstance(value, np.ndarray):
            assert value.shape == it.shape
            if self._dtype is not None:
                value = value.astype(self._dtype)
            for i, pos in it.iter_with_indices():
                self.set_value(pos, value[i])
        else:
            for pos in it:
                self.set_value(pos, value)

    def _set_positions(self, pos: np.ndarray, value: Union[V, Sequence]):
        if isinstance(pos, list):
            if not pos:
                return  # No Op
        pos = np.asarray(pos, dtype=int)
        if pos.shape == (3,):
            self.set_value(pos, value)
        else:
            assert pos.ndim == 2 and pos.shape[1] == 3, f"shape={pos.shape}"
            if isinstance(value, (list, tuple, np.ndarray)):
                assert len(pos) == len(value)
                for p, v in zip(pos, value):
                    self.set_value(p, v)
            else:
                upos = np.unique(pos, axis=0)
                for p in upos:
                    self.set_value(p, value)

    def _set_chunks(self, mask: "ChunkGrid", value: Union[V, np.ndarray, Chunk[V], "ChunkGrid[V]"]):
        assert self._chunk_size == mask._chunk_size
        # Method cache (prevent lookup in loop)
        __grid_ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index
        if isinstance(value, ChunkGrid):
            for m in mask.chunks:
                val = __grid_ensure_chunk_at_index(value, m.index, insert=False)
                __grid_ensure_chunk_at_index(self, m.index)[m] = val
        else:
            for m in mask.chunks:
                __grid_ensure_chunk_at_index(self, m.index)[m] = value

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

    def cleanup(self, remove=False):
        for chunk in self.chunks:
            chunk.cleanup()
        if remove:
            for chunk in list(self.chunks):
                if not chunk.any():
                    del self.chunks[chunk.index]
        return self

    def pad_chunks(self, width: int = 1):
        visited: Set[ChunkIndex] = set()
        for s in range(0, width):
            extra: Set[ChunkIndex] = set(tuple(n) for i in self.chunks.keys()
                                         for f, n in self.iter_neighbors_indicies(i))
            extra = extra.difference(visited)
            for e in extra:
                self.ensure_chunk_at_index(e)
            visited.update(extra)

    def iter_hull(self) -> Iterator[Chunk[V]]:
        """Iter some of the outer chunks that represent the hull around all chunks"""
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

    # Operators

    def apply(self, func: Callable[[Union[Chunk[V], V]], Union[Chunk[M], M]],
              dtype: Optional[Type[M]] = None, inplace=False, ) -> "ChunkGrid[M]":
        dtype = self.__dtype(dtype)
        # Inplace selection
        new_grid = self if inplace else self.copy(empty=True, dtype=dtype)

        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert

        for i, a in self.chunks.items():
            new_chunk = func(a)
            assert isinstance(new_chunk, Chunk)
            __new_grid_chunks_insert(i, new_chunk)

        if dtype is not None:
            new_grid._dtype = dtype
            new_grid._fill_value = dtype.type(func(self._fill_value))
        return new_grid

    def outer_join(self, rhs, func: Callable[[Union[Chunk[V], V], Union[Chunk[V], V]], Union[Chunk[M], M]],
                   dtype: Optional[Type[M]] = None, inplace=False) -> "ChunkGrid[M]":
        dtype = self.__dtype(dtype)
        # Inplace selection
        new_grid = self if inplace else self.copy(empty=True, dtype=dtype)

        if inplace:
            # Update data type
            for c in new_grid.chunks.values():
                c._dtype = dtype

        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert

        if isinstance(rhs, ChunkGrid):
            assert new_grid._chunk_size == rhs._chunk_size
            indices = set(self.chunks.keys())
            indices.update(rhs.chunks.keys())

            # Method cache (prevent lookup in loop)
            _ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index

            for i in indices:
                a = _ensure_chunk_at_index(self, i, insert=False)
                b = _ensure_chunk_at_index(rhs, i, insert=False)
                new_chunk = func(a, b)
                assert isinstance(new_chunk, Chunk)
                __new_grid_chunks_insert(i, new_chunk)
        else:
            for i, a in self.chunks.items():
                new_chunk = func(a, rhs)
                assert isinstance(new_chunk, Chunk)
                __new_grid_chunks_insert(i, new_chunk)

        # Update dtype and if possible the default fill value
        # noinspection DuplicatedCode
        if dtype is not None:
            new_grid._dtype = dtype
            try:
                if isinstance(rhs, ChunkGrid):
                    new_grid._fill_value = func(self._fill_value, rhs._fill_value)
                else:
                    new_grid._fill_value = func(self._fill_value, rhs)
            except Exception:
                try:
                    new_grid._fill_value = dtype.type(self._fill_value)
                except Exception:
                    pass
        return new_grid

    # Comparison Operator

    def __eq__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.eq, dtype=np.bool8)

    def __ne__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.ne, dtype=np.bool8)

    def __lt__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.lt, dtype=np.bool8)

    def __le__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.le, dtype=np.bool8)

    def __gt__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.gt, dtype=np.bool8)

    def __ge__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.ge, dtype=np.bool8)

    eq = __eq__
    ne = __ne__
    lt = __lt__
    le = __le__
    gt = __gt__
    ge = __ge__

    equals = __eq__

    # Single Operator

    def __abs__(self) -> "ChunkGrid[V]":
        return self.apply(operator.abs, dtype=self._dtype)

    def __invert__(self) -> "ChunkGrid[V]":
        return self.apply(operator.inv, dtype=self._dtype)

    def __neg__(self) -> "ChunkGrid[V]":
        return self.apply(operator.neg, dtype=self._dtype)

    abs = __abs__
    invert = __invert__
    neg = __neg__

    # Logic Operator

    def __and__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.and_, dtype=self._dtype)

    def __or__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.or_, dtype=self._dtype)

    def __xor__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.xor, dtype=self._dtype)

    def __iand__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.iand, dtype=self._dtype, inplace=True)

    def __ior__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.ior, dtype=self._dtype, inplace=True)

    def __ixor__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.ixor, dtype=self._dtype, inplace=True)

    and_ = __and__
    or_ = __or__
    xor = __xor__

    # Math Operator

    def __add__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.add)

    def __sub__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.sub)

    def __mul__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.mul)

    def __matmul__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.matmul)

    def __mod__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.mod)

    def __pow__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.pow)

    def __floordiv__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.floordiv)

    def __iadd__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.iadd, inplace=True)

    def __isub__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.isub, inplace=True)

    def __imul__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.imul, inplace=True)

    def __imatmul__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.imatmul, inplace=True)

    def __imod__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.imod, inplace=True)

    def __ifloordiv__(self, rhs) -> "ChunkGrid[V]":
        return self.outer_join(rhs, func=operator.ifloordiv, inplace=True)

    # TrueDiv Operator

    def __truediv__(self, rhs: Union["ChunkGrid[V]", np.ndarray, V]) -> "ChunkGrid[float]":
        return self.outer_join(rhs, func=operator.truediv, dtype=np.float)

    def __itruediv__(self, rhs: Union["ChunkGrid[V]", np.ndarray, V]) -> "ChunkGrid[float]":
        return self.outer_join(rhs, func=operator.itruediv, dtype=np.float, inplace=True)

    # Reflected Operators
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    __rtruediv__ = __truediv__
    __rfloordiv__ = __floordiv__
    __rmod__ = __mod__
    __rpow__ = __pow__
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    def sum(self, dtype: Optional[Type[M]] = None) -> M:
        return sum(c.sum() for c in self.chunks)

    def _correlate_filter(self, func, kernel: np.ndarray, fill_value: Optional[M] = None, fast=False,
                          **kwargs) -> "ChunkGrid[M]":
        fill_value = fill_value if fill_value is not None else self._fill_value
        new_grid = ChunkGrid(self._chunk_size, dtype=self._dtype, fill_value=fill_value)
        for c in self.chunks:
            if fast and not c.any():
                continue
            new_chunk = c._correlate_filter(func, kernel, self, fill_value=fill_value, **kwargs)
            new_grid.chunks.insert(c.index, new_chunk)
        new_grid.cleanup(remove=True)
        return new_grid

    def correlate(self, kernel: np.ndarray, fill_value: Optional[M] = None, fast=False) -> "ChunkGrid[M]":
        fill_value = fill_value if fill_value is not None else self._fill_value
        return self._correlate_filter(ndimage.correlate, kernel, fill_value, fast, mode='constant', cval=fill_value)

    def convolve(self, kernel: np.ndarray, fill_value: Optional[M] = None, fast=False) -> "ChunkGrid[M]":
        fill_value = fill_value if fill_value is not None else self._fill_value
        return self._correlate_filter(ndimage.convolve, kernel, fill_value, fast, mode='constant', cval=fill_value)

    @classmethod
    def stack(cls, grids: Sequence["ChunkGrid[V]"], fill_value=None) -> "ChunkGrid[np.ndarray]":
        assert len(grids) > 0
        if len(grids) == 1:
            return grids[0]
        # Check that grid size matches!
        chunk_size = grids[0].chunk_size
        assert all(chunk_size == g.chunk_size for g in grids)
        indices = set(k for g in grids for k in g.chunks.keys())
        dtypes = [g.dtype for g in grids]
        if all(t == dtypes[0] for t in dtypes):
            assert dtypes[0].base is not np.void
            dtype = np.dtype((dtypes[0].base, (len(grids),)))
            fill_value = fill_value if fill_value is not None else np.zeros(1, dtype)
        else:
            raise ValueError("Mixed data type grids not supported!")

        new_grid = ChunkGrid(chunk_size, dtype=dtype, fill_value=fill_value)
        for ind in indices:
            new_grid.chunks.insert(ind, Chunk._stack([g.ensure_chunk_at_index(ind, insert=False) for g in grids],
                                                     dtype=dtype, fill_value=fill_value))
        return new_grid
