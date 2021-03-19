import unittest

import numpy as np

from reconstruction.data.chunks import Chunk


class TestChunkOperator(unittest.TestCase):
    def test_eq_1(self):
        a = Chunk((0, 0, 0), 2, dtype=int, fill_value=0)
        b = Chunk((0, 0, 0), 2, dtype=int, fill_value=0)

        res = a == b
        self.assertIs(res.dtype.type, np.bool8)
        result = res.to_array()

        expected = np.ones((2, 2, 2), dtype=bool)
        self.assertEqual(str(expected), str(result))

    def test_eq_2(self):
        a = Chunk((0, 0, 0), 2, dtype=int, fill_value=0)
        b = Chunk((0, 0, 0), 2, dtype=int, fill_value=0)

        a.set_pos((0, 0, 0), 1)
        b.set_pos((0, 0, 0), 1)

        a.set_pos((1, 1, 1), 1)
        b.set_pos((0, 0, 1), 1)

        res = a == b
        self.assertIs(res.dtype.type, np.bool8)
        result = res.to_array()

        expected = np.ones((2, 2, 2), dtype=bool)
        expected[1, 1, 1] = False
        expected[0, 0, 1] = False
        self.assertEqual(str(expected), str(result))

    def test_eq_3(self):
        a = Chunk((0, 0, 0), 2, dtype=np.bool8, fill_value=0)
        b = Chunk((0, 0, 0), 2, dtype=np.bool8, fill_value=0)

        a.set_pos((0, 0, 0), 1)
        b.set_pos((0, 0, 0), 1)

        a.set_pos((1, 1, 1), 1)
        b.set_pos((0, 0, 1), 1)

        res = a == b
        self.assertIs(res.dtype.type, np.bool8)
        result = res.to_array()

        expected = np.ones((2, 2, 2), dtype=bool)
        expected[1, 1, 1] = False
        expected[0, 0, 1] = False
        self.assertEqual(str(expected), str(result))
