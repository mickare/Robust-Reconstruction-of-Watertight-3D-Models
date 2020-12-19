import enum
from typing import Union, Tuple, Iterator, Optional, Generic, TypeVar

import numpy as np

from data.index_dict import IndexDict, Index
from mathlib import Vec3i

V = TypeVar('V')
ChunkIndex = Index


class ChunkType(enum.Enum):
    EMPTY = 0
    FILL = 1
    ARRAY = 2


class ChunkFace(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    TOP = 4
    BOTTOM = 5

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


class Chunk(Generic[V]):
    def __init__(self, index: Vec3i, size: int, empty_value: Optional[T] = None):
        self._index: Vec3i = np.asarray(index, dtype=np.int)
        self._size = size
        self._empty_value = empty_value
        self._type = ChunkType.EMPTY
        self._value: Union[None, V, np.ndarray] = None

    @property
    def index(self) -> Vec3i:
        return self._index

    @property
    def type(self) -> ChunkType:
        return self._type

    @property
    def value(self) -> Union[None, V, np.ndarray]:
        return self._value

    @property
    def size(self) -> int:
        return self._size

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._size, self._size, self._size

    def empty(self) -> bool:
        return self._type is ChunkType.EMPTY

    def inner(self, pos: Vec3i) -> np.ndarray:
        return np.asarray(pos, dtype=np.int) % self._size

    def set_pos(self, pos: Vec3i, value: V):
        inner = self.inner(pos)
        arr = self.to_array()
        arr[tuple(inner)] = value
        self.set_array(arr)

    def set_fill(self, value: V):
        self._value = value
        self._type = ChunkType.FILL

    def clear(self):
        self._value = None
        self._type = ChunkType.EMPTY

    def set_array(self, value: np.ndarray):
        assert value.shape == self.shape
        self._value = value
        self._type = ChunkType.ARRAY

    def is_filled(self, *, value: Optional[V] = None, logic: Optional[bool] = None):
        if self._value is None:
            return value is None and logic is None
        else:
            if logic is not None:
                return bool(self._value) == logic
            elif value is None and isinstance(self._value, bool):
                return self._value
            else:
                return self._value == value

    def mask(self, *, value: Optional[V] = None, logic: Optional[bool] = None) -> np.ndarray:
        assert not (value is not None and logic is not None)  # Args value and logic cannot be used at the same time
        if self._type == ChunkType.EMPTY:
            return np.full(self.shape, False, dtype=np.bool)
        elif self._type == ChunkType.FILL:
            return np.full(self.shape, self.is_filled(value=value, logic=logic), dtype=np.bool)
        elif self._type == ChunkType.ARRAY:
            if logic is not None:
                return self._value.astype(np.bool) == logic
            elif value is None and self._value.dtype != np.object:
                return self._value.astype(np.bool)
            else:
                return self._value == value
        else:
            raise RuntimeError(f"Unexpected chunk type {self._type}")

    def to_array(self):
        if self._type == ChunkType.EMPTY:
            return np.full(self.shape, self._empty_value)
        elif self._type == ChunkType.FILL:
            return np.full(self.shape, self._value)
        elif self._type == ChunkType.ARRAY:
            return self._value
        else:
            raise RuntimeError(f"Unexpected chunk type {self._type}")


class ChunkGrid(Generic[V]):
    def __init__(self, chunk_size: int = 8, empty_value: Optional[T] = None):
        assert chunk_size > 0
        self._chunk_size = chunk_size
        self.chunks: IndexDict[Chunk[V]] = IndexDict()
        self._empty_value = empty_value

    def _new_chunk_factory(self, index: Index):
        return Chunk(index, self._chunk_size, self._empty_value)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        s = self._chunk_size
        return s, s, s

    @property
    def empty_value(self):
        return self._empty_value

    def chunk_index(self, pos: Vec3i):
        return np.asarray(pos, dtype=np.int) // self._chunk_size

    def chunk_at_pos(self, pos: Vec3i) -> Optional[Chunk[V]]:
        return self.chunks.get(self.chunk_index(pos))

    def ensure_chunk_at_index(self, index: ChunkIndex) -> Chunk[V]:
        return self.chunks.create_if_absent(index, self._new_chunk_factory)

    def ensure_chunk_at_pos(self, pos: Vec3i) -> Chunk[V]:
        return self.ensure_chunk_at_index(self.chunk_index(pos))

    def empty_mask(self, default=False) -> np.ndarray:
        return np.full(self.chunk_shape, default, dtype=np.bool)

    def iter_neighbors_indicies(self, index: ChunkIndex) -> Iterator[Tuple[ChunkFace, Index]]:
        yield from ((f, np.add(index, f.direction)) for f in ChunkFace)

    def iter_neighbors(self, index: ChunkIndex, flatten=False) -> Iterator[Tuple[ChunkFace, Optional["Chunk"]]]:
        if flatten:
            yield from ((f, c) for f, c in self.iter_neighbors(index, False) if c is not None)
        else:
            yield from ((f, self.chunks.get(i, None)) for f, i in self.iter_neighbors_indicies(index))
