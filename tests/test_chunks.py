import operator
import unittest
from typing import Type

import numpy as np

from data.chunks import ChunkGrid, ChunkFace


class TestChunkSetter(unittest.TestCase):
    def test_set_pos(self):
        a = ChunkGrid(2, bool, False)
        a.set_pos((0, 0, 0), True)
        a.set_pos((2, 2, 2), True)

        self.assertFalse(a.chunks[0, 0, 0].is_filled())

        result = a.to_dense()

        expected = np.zeros((4, 4, 4), dtype=bool)
        expected[0, 0, 0] = True
        expected[2, 2, 2] = True

        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Not equal: {result}\n-------\n{expected}")


class TestChunkOperator(unittest.TestCase):

    def test_eq_1(self):
        a = ChunkGrid(2, bool, False)
        b = ChunkGrid(2, bool, False)
        a.set_pos((0, 0, 0), True)
        result = (a == b).to_dense()
        self.assertIsInstance((a == b), ChunkGrid)

        expected = np.ones((2, 2, 2), dtype=bool)
        expected[0, 0, 0] = False

        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Failure! \n{result}\n-------\n{expected}")

    def test_eq_2(self):
        a = ChunkGrid(2, bool, False)
        b = ChunkGrid(2, bool, False)

        a.set_pos((0, 0, 0), True)
        a.set_pos((2, 2, 2), True)

        result = (a == b).to_dense()
        self.assertIsInstance((a == b), ChunkGrid)

        expected = np.ones((4, 4, 4), dtype=bool)
        expected[0, 0, 0] = False
        expected[2, 2, 2] = False

        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Failure! \n{result}\n-------\n{expected}")

    def _test_operator1_bool(self, op):
        a = ChunkGrid(2, bool, False)

        a.set_pos((0, 0, 0), True)
        a.set_pos((0, 0, 1), True)
        a.set_pos((0, 1, 0), False)

        expected = op(a.to_dense())
        res = op(a)
        self.assertIsInstance(res, ChunkGrid)
        result = res.to_dense()
        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Failure {op}! \n{result}\n-------\n{expected}")

    def _test_operator1_int(self, op):
        a = ChunkGrid(2, int, -1)

        a.set_pos((0, 0, 0), 0)
        a.set_pos((0, 0, 1), 2)
        a.set_pos((0, 1, 0), -2)

        expected = op(a.to_dense())
        res = op(a)
        self.assertIsInstance(res, ChunkGrid)
        result = res.to_dense()
        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Failure {op}! \n{result}\n-------\n{expected}")

    def test_operators1(self):
        self._test_operator1_bool(operator.inv)

        self._test_operator1_int(operator.abs)
        self._test_operator1_int(operator.inv)
        self._test_operator1_int(operator.neg)

    def _test_inplace_modified(self, op, a: ChunkGrid, a0: ChunkGrid, b: ChunkGrid, b0: ChunkGrid, inplace: bool):
        # Test if inplace (or not) is working intended
        ad = a.to_dense()
        a0d = a0.to_dense()
        bd = b.to_dense()
        b0d = b0.to_dense()
        if inplace:  # Inplace modification - change
            opa = ad == op(ad, bd)
            opb = bd == b0d
        else:  # Not inplace modification - no change
            opa = ad == a0d
            opb = bd == b0d
        self.assertTrue(np.all(opa), f"Failure-Inplace-{inplace} {op}! \n{ad}\n-------\n{a0d}")
        self.assertTrue(np.all(opb), f"Failure-Inplace-{inplace} {op}! \n{bd}\n-------\n{b0d}")

    def _test_operator2_bool(self, op, inplace=False):
        a = ChunkGrid(2, bool, False)
        b = ChunkGrid(2, bool, True)

        a.set_pos((0, 0, 0), True)
        a.set_pos((0, 0, 1), True)

        b.set_pos((0, 0, 0), True)
        b.set_pos((0, 0, 1), False)
        b.set_pos((0, 1, 0), False)
        b.set_pos((0, 1, 1), False)

        a0 = a.copy()
        b0 = b.copy()
        expected = op(a.to_dense(), b.to_dense())
        res = op(a, b)
        self.assertIsInstance(res, ChunkGrid)
        result = res.to_dense()
        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Failure {op}! \n{result}\n-------\n{expected}")
        self._test_inplace_modified(op, a, a0, b, b0, inplace)

    def _test_operator2_int(self, op, b0=0, dtype: Type = int, inplace=False):
        a = ChunkGrid(2, dtype, 0)
        b = ChunkGrid(2, dtype, 1)

        a.set_pos((0, 0, 0), 1)
        a.set_pos((0, 0, 1), 1)

        b.set_pos((0, 0, 0), 1 + b0)
        b.set_pos((0, 1, 0), 0 + b0)
        b.set_pos((0, 1, 1), 0 + b0)

        a0 = a.copy()
        b0 = b.copy()
        expected = op(a.to_dense(), b.to_dense())
        res = op(a, b)
        self.assertIsInstance(res, ChunkGrid)
        result = res.to_dense()
        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Failure {op}! \n{result}\n-------\n{expected}")
        self._test_inplace_modified(op, a, a0, b, b0, inplace)

    def test_operators2(self):
        self._test_operator2_bool(operator.eq)
        self._test_operator2_bool(operator.ne)

        self._test_operator2_int(operator.eq)
        self._test_operator2_int(operator.ne)

        self._test_operator2_bool(operator.and_)
        self._test_operator2_bool(operator.or_)
        self._test_operator2_bool(operator.xor)

        self._test_operator2_int(operator.add)
        self._test_operator2_int(operator.floordiv, 2)
        self._test_operator2_int(operator.mod)
        self._test_operator2_int(operator.mul)
        self._test_operator2_int(operator.matmul)
        self._test_operator2_int(operator.sub)
        self._test_operator2_int(operator.truediv, 2, dtype=float)

        self._test_operator2_bool(operator.iand, inplace=True)
        self._test_operator2_bool(operator.ior, inplace=True)
        self._test_operator2_bool(operator.ixor, inplace=True)

        self._test_operator2_int(operator.iadd, inplace=True)
        self._test_operator2_int(operator.isub, inplace=True)
        self._test_operator2_int(operator.imul, inplace=True)
        self._test_operator2_int(operator.itruediv, 2, dtype=float, inplace=True)
        self._test_operator2_int(operator.ifloordiv, 2, inplace=True)
        self._test_operator2_int(operator.imod, inplace=True)

    # def test_and(self):
    #     a = ChunkGrid(2, bool, False)
    #     b = ChunkGrid(2, bool, False)
    #     a.set_pos((0, 0, 0), True)
    #     a.set_pos((0, 0, 1), True)
    #     b.set_pos((0, 0, 0), True)
    #     b.set_pos((0, 1, 0), True)
    #     result = (a & b).to_dense()
    #     self.assertIsInstance((a & b), ChunkGrid)
    #
    #     expected = np.zeros((2, 2, 2), dtype=bool)
    #     expected[0, 0, 0] = True
    #
    #     assert result.shape == expected.shape
    #     self.assertTrue(np.all(result == expected), f"Failure! \n{result}\n-------\n{expected}")
    #
    # def test_ne(self):
    #     a = ChunkGrid(2, bool, False)
    #     b = ChunkGrid(2, bool, False)
    #     a.set_pos((0, 0, 0), True)
    #     a.set_pos((0, 0, 1), True)
    #     b.set_pos((0, 0, 0), True)
    #     b.set_pos((0, 1, 0), True)
    #     result = (a != b).to_dense()
    #     self.assertIsInstance((a != b), ChunkGrid)
    #
    #     expected = np.zeros((2, 2, 2), dtype=bool)
    #     expected[0, 0, 1] = True
    #     expected[0, 1, 0] = True
    #
    #     assert result.shape == expected.shape
    #     self.assertTrue(np.all(result == expected), f"Failure! \n{result}\n-------\n{expected}")

    def test_padding(self):
        grid = ChunkGrid(2, int, -1)
        grid.ensure_chunk_at_index((0, 0, 0)).set_fill(1)
        grid.ensure_chunk_at_index((0, 0, 1)).set_fill(2)
        grid.ensure_chunk_at_index((0, 1, 0)).set_fill(3)
        grid.ensure_chunk_at_index((0, 1, 1)).set_fill(4)

        t = grid.chunks[(0, 0, 1)]
        t.set_array(np.array([
            [(111, 112), (121, 122)],
            [(211, 212), (221, 222)]
        ]))
        expected1 = t.to_array()[:, :, 0]

        c = grid.chunks.get((0, 0, 0))
        pad = c.padding(grid, 1)
        actual = pad

        expected = np.ones((4, 4, 4), int) * -1
        expected[1:3, 1:3, 1:3] = 1
        expected[1:-1, 1:-1, -1] = expected1
        expected[1:-1, -1, 1:-1] = 3

        assert actual.shape == expected.shape, f"Shape! \n{actual.shape}\n-------\n{expected.shape}"
        # self.assertTrue(np.all(actual == expected), f"Failure! \n{actual}\n-------\n{expected}")
        self.assertEqual(str(expected), str(actual))

    def test_face_slicing(self):
        s = slice(None)
        s0 = -1
        s1 = 0
        self.assertEqual(ChunkFace.NORTH.slice(), (s0, s, s))
        self.assertEqual(ChunkFace.SOUTH.slice(), (s1, s, s))
        self.assertEqual(ChunkFace.TOP.slice(), (s, s0, s))
        self.assertEqual(ChunkFace.BOTTOM.slice(), (s, s1, s))
        self.assertEqual(ChunkFace.EAST.slice(), (s, s, s0))
        self.assertEqual(ChunkFace.WEST.slice(), (s, s, s1))

        w = 1
        s0 = slice(- w, None)
        s1 = slice(None, w)
        self.assertEqual(ChunkFace.NORTH.slice(w), (s0, s, s))
        self.assertEqual(ChunkFace.SOUTH.slice(w), (s1, s, s))
        self.assertEqual(ChunkFace.TOP.slice(w), (s, s0, s))
        self.assertEqual(ChunkFace.BOTTOM.slice(w), (s, s1, s))
        self.assertEqual(ChunkFace.EAST.slice(w), (s, s, s0))
        self.assertEqual(ChunkFace.WEST.slice(w), (s, s, s1))

        w = 2
        s0 = slice(- w, None)
        s1 = slice(None, w)
        self.assertEqual(ChunkFace.NORTH.slice(w), (s0, s, s))
        self.assertEqual(ChunkFace.SOUTH.slice(w), (s1, s, s))
        self.assertEqual(ChunkFace.TOP.slice(w), (s, s0, s))
        self.assertEqual(ChunkFace.BOTTOM.slice(w), (s, s1, s))
        self.assertEqual(ChunkFace.EAST.slice(w), (s, s, s0))
        self.assertEqual(ChunkFace.WEST.slice(w), (s, s, s1))

        other = 1
        w = 1
        s = slice(other, -other)
        s0 = slice(- w, None)
        s1 = slice(None, w)
        self.assertEqual(ChunkFace.NORTH.slice(w, other=s), (s0, s, s))
        self.assertEqual(ChunkFace.SOUTH.slice(w, other=s), (s1, s, s))
        self.assertEqual(ChunkFace.TOP.slice(w, other=s), (s, s0, s))
        self.assertEqual(ChunkFace.BOTTOM.slice(w, other=s), (s, s1, s))
        self.assertEqual(ChunkFace.EAST.slice(w, other=s), (s, s, s0))
        self.assertEqual(ChunkFace.WEST.slice(w, other=s), (s, s, s1))
