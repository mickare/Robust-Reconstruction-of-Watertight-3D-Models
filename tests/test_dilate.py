import unittest

import numpy as np

from data.chunks import ChunkGrid
from filters.dilate import dilate


class TestDilate(unittest.TestCase):
    def test_filled(self):
        a = ChunkGrid(2, bool, False)
        a.ensure_chunk_at_index((0, 0, 0)).set_fill(True)

        res = dilate(a, steps=1)
        self.assertEqual(7, len(res.chunks))
        c0 = res.chunks.get((0, 0, 0))
        self.assertIsNotNone(c0)
        self.assertTrue(c0.is_filled())
        self.assertTrue(c0.value)

        expected = np.zeros((6, 6, 6), dtype=bool)
        expected[2:-2, 2:-2, 1:-1] = True
        expected[2:-2, 1:-1, 2:-2] = True
        expected[1:-1, 2:-2, 2:-2] = True
        self.assertEqual(str(expected), str(res.to_dense()))

    def test_single_positive(self):
        a = ChunkGrid(2, bool, False)
        a.set_value((1, 1, 1), True)

        res = dilate(a, steps=1)
        self.assertEqual(4, len(res.chunks))
        c0 = res.chunks.get((0, 0, 0))
        self.assertIsNotNone(c0)
        self.assertFalse(c0.is_filled())

        expected = np.zeros((4, 4, 4), dtype=bool)
        expected[1, 1, 0:3] = True
        expected[1, 0:3, 1] = True
        expected[0:3, 1, 1] = True

        self.assertEqual(str(expected), str(res.to_dense()))

    def test_single_negative(self):
        a = ChunkGrid(2, bool, False)
        a.set_value((0, 0, 0), True)

        res = dilate(a, steps=1)
        self.assertEqual(4, len(res.chunks))
        c0 = res.chunks.get((0, 0, 0))
        self.assertIsNotNone(c0)
        self.assertFalse(c0.is_filled())

        expected = np.zeros((4, 4, 4), dtype=bool)
        expected[2, 2, 1:4] = True
        expected[2, 1:4, 2] = True
        expected[1:4, 2, 2] = True

        self.assertEqual(str(expected), str(res.to_dense()))
