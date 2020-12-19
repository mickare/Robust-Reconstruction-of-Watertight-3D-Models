from typing import Dict, Union, Tuple, Iterator, Optional, Iterable, Sequence, Generic, TypeVar, Callable, List, \
    ItemsView, KeysView

import numpy as np

from data.data_utils import MinMaxCheck, SliceIterator

T = TypeVar('T')
Index = Tuple[int, int, int]
IndexUnion = Union[Index, Sequence[int], np.ndarray]


class IndexDict(Generic[T]):
    def __init__(self) -> None:
        self._data: Dict[Index, T] = dict()
        self._minmax = MinMaxCheck()

    def clear(self):
        self._data.clear()
        self._minmax.clear()

    def __delitem__(self, key):
        index = self.index(key)
        del self._data[index]
        self._minmax.set_dirty()

    def pop(self, index, default=None) -> Optional[T]:
        l = len(self._data)
        c = self._data.pop(index, default)
        if l != len(self._data):
            self._minmax.set_dirty()
        return c

    def index(self, item: IndexUnion) -> Index:
        item = np.asarray(item, dtype=np.int)
        assert item.shape == (3,)
        return tuple(item)

    def get(self, index: Index, default=None) -> Optional[T]:
        return self._data.get(self.index(index), default)

    def __getitem_from_numpy(self, item: np.ndarray, ignore_empty=True) -> Union[T, List[T]]:
        item = np.asarray(item, dtype=np.int)
        if item.shape == (3,):
            return self._data[tuple(item)]
        else:
            assert item.ndim == 2 and item.shape[1] == 3
            if ignore_empty:
                return [d for d in (self._data.get(tuple(i), None) for i in item) if d is not None]
            else:
                return [self._data[tuple(i)] for i in item]

    def sliced(self, x: Union[int, slice, None] = None, y: Union[int, slice, None] = None,
               z: Union[int, slice, None] = None, ignore_empty=True) -> Iterator[T]:
        if not self._data:
            return
        if x is None and y is None and z is None:
            yield from self._data.values()
        else:
            min, max = self._minmax.safe(self._data.keys)
            it_u = SliceIterator(x, (min[0], max[0]))
            it_v = SliceIterator(y, (min[1], max[1]))
            it_w = SliceIterator(z, (min[2], max[2]))
            if ignore_empty:
                for u in it_u:
                    for v in it_v:
                        for w in it_w:
                            key = (u, v, w)
                            if key in self._data:
                                yield self._data[key]
            else:
                for u in it_u:
                    for v in it_v:
                        for w in it_w:
                            yield self._data.get((u, v, w))

    def __getitem__(self, item: Union[IndexUnion, slice, Tuple[slice, ...]]) -> Union[T, List[T]]:
        if isinstance(item, slice):
            return list(self.sliced(item))
        if isinstance(item, tuple):
            try:
                index = np.asarray(item, dtype=np.int)
                if index.shape == (3,):
                    return self._data[self.index(item)]
                raise KeyError(f"invalid key {item}")
            except TypeError:
                pass
            if 0 < len(item) <= 3 and any(isinstance(i, slice) for i in item):
                return list(self.sliced(*item))
            raise KeyError(f"invalid key {item}")
        elif isinstance(item, (list, np.ndarray)):
            return self.__getitem_from_numpy(np.array(item, dtype=int))
        else:
            raise KeyError(f"invalid key {item}")

    def set(self, index: Index, value: T):
        index = self.index(index)
        self._data[index] = value
        self._minmax.add(index)

    def __setitem__(self, key: Index, value: T):
        index = self.index(key)
        self._data[index] = value
        self._minmax.add(index)

    def __contains__(self, item: Index) -> bool:
        return self.index(item) in self._data

    def setdefault(self, index: Index, default: T) -> T:
        index = self.index(index)
        c = self._data.get(index, None)
        if c is None:
            self._data[index] = default
            self._minmax.add(index)
            return default
        return c

    def create_if_absent(self, index: Index, factory: Callable[[Index], T]) -> T:
        index = self.index(index)
        c = self._data.get(index, None)
        if c is None:
            c = factory(index)
            self._data[index] = c
            self._minmax.add(index)
        return c

    def __len__(self):
        return len(self._data)

    def items(self) -> ItemsView[Index, T]:
        return self._data.items()

    def keys(self) -> KeysView[Index]:
        return self._data.keys()

    def __iter__(self) -> Iterable[T]:
        return iter(self._data.values())
