import operator
import unittest
from typing import Type

import numpy as np

from data.chunks import ChunkGrid
from data.faces import ChunkFace


class TestChunkGridSetter(unittest.TestCase):
    def test_set_pos(self):
        a = ChunkGrid(2, bool, False)
        a.set_value((0, 0, 0), True)
        a.set_value((2, 2, 2), True)

        self.assertFalse(a.chunks[0, 0, 0].is_filled())

        result = a.to_dense()

        expected = np.zeros((4, 4, 4), dtype=bool)
        expected[0, 0, 0] = True
        expected[2, 2, 2] = True

        assert result.shape == expected.shape
        self.assertEqual(str(expected), str(result))

    def test_mask_1(self):
        a = ChunkGrid(2, int, 0)

        mask = ChunkGrid(2, bool, False)
        mask[1:3, 1:3, 1:3] = True
        self.assertEqual((4, 4, 4), mask.to_dense().shape)
        a[mask] = 3

        expected = np.zeros((4, 4, 4), dtype=int)
        expected[1:3, 1:3, 1:3] = 3

        self.assertEqual(str(expected), str(a.to_dense()))

        tmp = a == 3
        self.assertEqual(str(mask.to_dense()), str(tmp.to_dense()))

    def test_mask_2(self):
        a = ChunkGrid(2, bool, False)
        a[0, 0, 0] = True
        a.ensure_chunk_at_index((1, 1, 1)).set_fill(True)
        a.set_value((2, 1, 1), True)

        b = a.astype(np.int8)

        adense = a.to_dense()
        bdense = b.to_dense()
        self.assertEqual(adense.shape, bdense.shape)
        self.assertEqual(str(adense.astype(np.int8)), str(bdense))

        mask = ChunkGrid(2, bool, False)
        mask.ensure_chunk_at_index((1, 0, 0)).set_fill(True)
        mask.ensure_chunk_at_index((0, 1, 0)).set_fill(True)
        mask.ensure_chunk_at_index((0, 0, 1))[0, 0, 0] = True

        b[mask] = 2
        bdense = b.to_dense()

        expected = np.zeros((4, 4, 4), dtype=np.int8)
        expected[0, 0, 0] = 1
        expected[2:4, 2:4, 2:4] = 1
        expected[2, 1, 1] = 1
        expected[2:4, 0:2, 0:2] = 2
        expected[0:2, 2:4, 0:2] = 2
        expected[0, 0, 2] = 2

        self.assertEqual(expected.shape, bdense.shape)
        self.assertEqual(str(expected), str(bdense))


class TestChunkGridOperator(unittest.TestCase):

    def test_eq_1(self):
        a = ChunkGrid(2, bool, False)
        b = ChunkGrid(2, bool, False)
        a.set_value((0, 0, 0), True)
        result = (a == b).to_dense()
        self.assertIsInstance((a == b), ChunkGrid)

        expected = np.ones((2, 2, 2), dtype=bool)
        expected[0, 0, 0] = False

        assert result.shape == expected.shape
        self.assertEqual(str(expected), str(result))

    def test_eq_2(self):
        a = ChunkGrid(2, bool, False)
        b = ChunkGrid(2, bool, False)

        a.set_value((0, 0, 0), True)
        a.set_value((2, 2, 2), True)

        comp = a == b
        self.assertIsInstance(comp, ChunkGrid)
        self.assertIs(comp.dtype.type, np.bool8)

        expected = np.ones((4, 4, 4), dtype=bool)
        expected[0, 0, 0] = False
        expected[2, 2, 2] = False

        result = comp.to_dense()
        assert result.shape == expected.shape
        self.assertEqual(str(expected), str(result))

    def test_eq_3(self):
        a = ChunkGrid(1, bool, False)
        b = ChunkGrid(1, bool, False)

        x = a == b
        self.assertEqual(0, len(x.chunks))
        self.assertTrue(x.fill_value)

    def _test_operator1_bool(self, op):
        a = ChunkGrid(2, bool, False)

        a.set_value((0, 0, 0), True)
        a.set_value((0, 0, 1), True)
        a.set_value((0, 1, 0), False)

        expected = op(a.to_dense())
        res = op(a)
        self.assertIsInstance(res, ChunkGrid)
        result = res.to_dense()
        assert result.shape == expected.shape
        self.assertTrue(np.all(result == expected), f"Failure {op}! \n{result}\n-------\n{expected}")

    def _test_operator1_int(self, op):
        a = ChunkGrid(2, int, -1)

        a.set_value((0, 0, 0), 0)
        a.set_value((0, 0, 1), 2)
        a.set_value((0, 1, 0), -2)

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

        a.set_value((0, 0, 0), True)
        a.set_value((0, 0, 1), True)

        b.set_value((0, 0, 0), True)
        b.set_value((0, 0, 1), False)
        b.set_value((0, 1, 0), False)
        b.set_value((0, 1, 1), False)

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

        a.set_value((0, 0, 0), 1)
        a.set_value((0, 0, 1), 1)

        b.set_value((0, 0, 0), 1 + b0)
        b.set_value((0, 1, 0), 0 + b0)
        b.set_value((0, 1, 1), 0 + b0)

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

        self.assertEqual(expected.shape, actual.shape)
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

    def test_reflected(self):
        grid = ChunkGrid(2, float, -1.0)
        grid.set_value((0, 0, 0), 0.5)
        grid.set_value((0, 0, 1), 1.0)

        result = (0 < grid) & (grid < 1.0)
        actual = result.to_dense()

        expected = np.zeros((2, 2, 2), dtype=bool)
        expected[0, 0, 0] = True

        self.assertEqual(str(expected), str(actual))

    def test_eq_and_combined(self):
        a = ChunkGrid(2, bool, False)
        a.set_value((0, 0, 0), True)
        a.set_value((0, 0, 1), True)
        a.set_value((0, 1, 1), True)

        b = ChunkGrid(2, int, 0)
        b.set_value((0, 0, 1), 1)
        b.set_value((0, 1, 1), 2)
        b.set_value((1, 1, 1), 1)

        tmp = (b == 1)
        self.assertEqual(1, len(tmp.chunks))
        self.assertFalse(next(iter(tmp.chunks)).is_filled())

        result = tmp & a
        self.assertEqual(1, len(result.chunks))
        self.assertFalse(next(iter(result.chunks)).is_filled())

        expected_tmp = np.zeros((2, 2, 2), dtype=bool)
        expected_tmp[0, 0, 1] = True
        expected_tmp[1, 1, 1] = True

        self.assertEqual(str(expected_tmp), str(tmp.to_dense()))

        expected = np.zeros((2, 2, 2), dtype=bool)
        expected[0, 0, 1] = True
        self.assertEqual(str(expected), str(result.to_dense()))

        values = list(tmp.items(mask=a))
        self.assertEqual(3, len(values))
        self.assertEqual(str([
            (np.array([0, 0, 0]), False),
            (np.array([0, 0, 1]), True),
            (np.array([0, 1, 1]), False)
        ]), str(values))

    def test_eq_int(self):
        a = ChunkGrid(2, int, 0)
        a.set_value((1, 1, 1), 1)
        a.ensure_chunk_at_index((1, 1, 1))

        result = a == 0
        dense = result.to_dense()

        expected = np.ones((4, 4, 4), dtype=bool)
        expected[1, 1, 1] = False

        self.assertEqual(expected.shape, dense.shape)
        self.assertEqual(str(expected), str(dense))
        self.assertTrue(result.any())
        self.assertFalse(result.all())


class TestChunkGridMethods(unittest.TestCase):
    def test_split(self):
        a = ChunkGrid(2, int, 0)
        a[0, 0, 0] = 1
        a[0, 0, 1] = 2
        a[0, 1, 0] = 3
        self.assertEqual(1, len(a.chunks))

        pos = np.array([
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ])

        b = a.split(2)
        self.assertEqual(8, len(b.chunks))

        offset = (0, 0, 0)
        for p in pos:
            self.assertEqual(1, b.get_value(p + offset))

        offset = (0, 0, 2)
        for p in pos:
            self.assertEqual(2, b.get_value(p + offset))

        offset = (0, 2, 0)
        for p in pos:
            self.assertEqual(3, b.get_value(p + offset))

        offset = (2, 0, 0)
        for p in pos:
            self.assertEqual(0, b.get_value(p + offset))

    def test_getitem(self):
        CS = 2
        shape = (CS * 3, CS * 3, CS * 3)
        soll = np.arange(shape[0] * shape[1] * shape[2]).reshape(shape)

        a = ChunkGrid(CS, int, -1)
        for u in range(shape[0] // CS):
            for v in range(shape[1] // CS):
                for w in range(shape[2] // CS):
                    x, y, z = u * CS, v * CS, w * CS
                    index = (u, v, w)
                    a.ensure_chunk_at_index(index).set_array(soll[x:x + CS, y:y + CS, z:z + CS])

        dense = a.to_dense()
        self.assertEqual(soll.shape, dense.shape)
        self.assertEqual(str(soll), str(dense))

        self.assertEqual(str(soll[1: shape[0] - 1, 1: shape[1] - 1, 1: shape[2] - 1]),
                         str(a[1: shape[0] - 1, 1: shape[1] - 1, 1: shape[2] - 1]))

    def test_getitem_offset(self):
        CS = 2
        shape = (CS * 3, CS * 3, CS * 3)
        soll = np.arange(shape[0] * shape[1] * shape[2]).reshape(shape)
        offset_chunk = (-1, 0, 1)

        a = ChunkGrid(CS, int, -1)
        for u in range(shape[0] // CS):
            for v in range(shape[1] // CS):
                for w in range(shape[2] // CS):
                    index = np.add((u, v, w), offset_chunk)
                    x, y, z = np.multiply((u, v, w), CS)
                    a.ensure_chunk_at_index(index).set_array(soll[x:x + CS, y:y + CS, z:z + CS])

        offset_voxel = np.multiply(offset_chunk, CS)
        dense, off = a.to_dense(return_offset=True)
        self.assertEqual(soll.shape, dense.shape)
        self.assertEqual(str(soll), str(dense))
        self.assertEqual(list(offset_voxel), list(off))

        ox, oy, oz = off
        self.assertEqual(str(soll[1: shape[0] - 1, 1: shape[1] - 1, 1: shape[2] - 1]),
                         str(a[1 + ox: shape[0] - 1 + ox, 1 + oy: shape[1] - 1 + oy, 1 + oz: shape[2] - 1 + oz]))

    def test__getitem_empty(self):
        a = ChunkGrid(2, int, -1)
        a.set_or_fill((0, 0, 0), 1)

        res = a[-1:1, -1:1, -1:1]

        expected = np.full((2, 2, 2), -1)
        expected[1, 1, 1] = 1

        self.assertEqual(str(expected), str(res))

    def test_special_dtype(self):

        t = np.dtype((np.int, (3,)))
        a = ChunkGrid(2, dtype=t, fill_value=np.zeros(3))

        a.set_value((0, 0, 0), np.ones(3))
        a.set_value((2, 2, 2), np.ones(3))
        dense = a.to_dense()

        expected = np.zeros((4, 4, 4), dtype=t)
        expected[0, 0, 0] = 1
        expected[2, 2, 2] = 1

        self.assertEqual(expected.shape, dense.shape)
        self.assertEqual(str(expected), str(dense))

    def test_get_block_1(self):
        a = ChunkGrid(2, int, 0)
        expected = np.zeros((6, 6, 6), dtype=int)
        for n, i in enumerate(np.ndindex(6, 6, 6)):
            pos = np.array(i, dtype=int) - 2
            a.set_value(pos, n + 1)
            expected[i] = n + 1

        block = a.get_block_at((0, 0, 0), (3, 3, 3), corners=True, edges=True)
        self.assertEqual(str(expected), str(a.block_to_array(block)))

    def test_get_block_2(self):
        a = ChunkGrid(2, int, 0)
        expected = np.zeros((6, 6, 6), dtype=int)
        for n, i in enumerate(np.ndindex(6, 6, 6)):
            pos = np.array(i, dtype=int) - 2
            a.set_value(pos, n + 1)
            s = np.sum(pos < 0) + np.sum(2 <= pos)
            if s >= 2:
                continue
            expected[i] = n + 1

        block = a.get_block_at((0, 0, 0), (3, 3, 3), corners=False, edges=False)
        self.assertEqual(str(expected), str(a.block_to_array(block)))

    def test_padding_1(self):
        a = ChunkGrid(2, int, 0)
        expected = np.zeros((6, 6, 6), dtype=int)
        for n, i in enumerate(np.ndindex(6, 6, 6)):
            pos = np.array(i, dtype=int) - 2
            a.set_value(pos, n + 1)
            expected[i] = n + 1

        data = a.padding_at((0, 0, 0), 2)
        self.assertEqual(expected.shape, data.shape)
        self.assertEqual(str(expected), str(data))

        expected2 = expected[1:-1, 1:-1, 1:-1]
        data2 = a.padding_at((0, 0, 0), 1)
        self.assertEqual(expected2.shape, data2.shape)
        self.assertEqual(str(expected2), str(data2))

        expected3_tmp = np.zeros((6, 6, 6), dtype=int)
        for n, i in enumerate(np.ndindex(6, 6, 6)):
            expected3_tmp[i] = n + 1
        expected3 = np.zeros((8, 8, 8), dtype=int)
        expected3[1:-1, 1:-1, 1:-1] = expected3_tmp
        data3 = a.padding_at((0, 0, 0), 3)
        self.assertEqual(expected3.shape, data3.shape)
        self.assertEqual(str(expected3), str(data3))

    def test_padding_2(self):
        a = ChunkGrid(2, int, 0)
        expected = np.zeros((6, 6, 6), dtype=int)
        for n, i in enumerate(np.ndindex(6, 6, 6)):
            pos = np.array(i, dtype=int) - 2
            a.set_value(pos, n + 1)
            s = np.sum(pos < 0) + np.sum(2 <= pos)
            if s >= 2:
                continue
            expected[i] = n + 1

        data = a.padding_at((0, 0, 0), 2, corners=False, edges=False)
        self.assertEqual(expected.shape, data.shape)
        self.assertEqual(str(expected), str(data))

        expected2 = expected[1:-1, 1:-1, 1:-1]
        data2 = a.padding_at((0, 0, 0), 1, corners=False, edges=False)
        self.assertEqual(expected2.shape, data2.shape)
        self.assertEqual(str(expected2), str(data2))

        expected3_tmp = np.zeros((6, 6, 6), dtype=int)
        for n, i in enumerate(np.ndindex(6, 6, 6)):
            expected3_tmp[i] = n + 1
        expected3 = np.zeros((8, 8, 8), dtype=int)
        expected3[1:-1, 1:-1, 1:-1] = expected3_tmp
        data3 = a.padding_at((0, 0, 0), 3, corners=False, edges=False)
        self.assertEqual(expected3.shape, data3.shape)
        self.assertEqual(str(expected3), str(data3))

    def test_padding_vec3_1(self):
        dtype = np.dtype((int, (3,)))
        a = ChunkGrid(2, dtype, np.zeros(3))
        a.set_value((0, 0, 2), np.ones(3) * 1)
        a.set_value((0, 2, 0), np.ones(3) * 2)
        a.set_value((2, 0, 0), np.ones(3) * 3)
        a.set_value((1, 1, 1), np.ones(3) * 4)

        a.set_value((2, 2, 2), np.ones(3) * -1)
        a.set_value((-1, -1, -1), np.ones(3) * -2)

        a.set_value((2, -1, -1), np.ones(3) * -3)
        a.set_value((-1, 2, -1), np.ones(3) * -4)
        a.set_value((-1, -1, 2), np.ones(3) * -5)

        a.set_value((2, 2, -1), np.ones(3) * -6)
        a.set_value((2, -1, 2), np.ones(3) * -7)
        a.set_value((-1, 2, 2), np.ones(3) * -8)

        pad = a.padding_at((0, 0, 0), 1, corners=False)

        expected = np.zeros((4, 4, 4, 3), dtype=int)
        expected[1, 1, 3] = 1
        expected[1, 3, 1] = 2
        expected[3, 1, 1] = 3
        expected[2, 2, 2] = 4

        self.assertEqual(expected.shape, pad.shape)
        self.assertEqual(str(expected), str(pad))

    def test_padding_vec3_2(self):
        a = ChunkGrid(2, int, 0)
        expected = np.zeros((6, 6, 6), dtype=int)

        for n, i in enumerate(np.ndindex(6, 6, 6)):
            pos = np.array(i, dtype=int) - 2
            value = (n + 1)
            a.set_value(pos, value)
            expected[i] = value

        data = a.padding_at((0, 0, 0), 2)
        self.assertEqual(expected.shape, data.shape)
        self.assertEqual(str(expected), str(data))

        expected2 = expected[1:-1, 1:-1, 1:-1]
        data2 = a.padding_at((0, 0, 0), 1)
        self.assertEqual(expected2.shape, data2.shape)
        self.assertEqual(str(expected2), str(data2))

    def test_padding_vec3_3(self):
        dtype = np.dtype((int, (3,)))
        a = ChunkGrid(2, dtype, np.zeros(3))
        expected = np.zeros((6, 6, 6, 3), dtype=int)

        for n, i in enumerate(np.ndindex(6, 6, 6)):
            pos = np.array(i, dtype=int) - 2
            value = np.array([1, 2, 3]) + (n + 1) * 3
            a.set_value(pos, value)
            expected[i] = value

        data = a.padding_at((0, 0, 0), 2, corners=True, edges=True)
        self.assertEqual(expected.shape, data.shape)
        self.assertEqual(str(expected), str(data))

        expected2 = expected[1:-1, 1:-1, 1:-1]
        data2 = a.padding_at((0, 0, 0), 1, corners=True, edges=True)
        self.assertEqual(expected2.shape, data2.shape)
        self.assertEqual(str(expected2), str(data2))

        expected3 = np.zeros((8, 8, 8, 3), dtype=int)
        expected3[1:-1, 1:-1, 1:-1] = expected
        data3 = a.padding_at((0, 0, 0), 3, corners=True, edges=True)
        self.assertEqual(expected3.shape, data3.shape)
        self.assertEqual(str(expected3), str(data3))

    def test_padding_vec3_4(self):
        dtype = np.dtype((int, (3,)))
        a = ChunkGrid(2, dtype, np.zeros(3))
        expected = np.zeros((6, 6, 6, 3), dtype=int)

        for n, i in enumerate(np.ndindex(6, 6, 6)):
            pos = np.array(i, dtype=int) - 2
            value = np.array([1, 2, 3]) + (n + 1) * 3
            a.set_value(pos, value)
            s = np.sum(pos < 0) + np.sum(2 <= pos)
            if s >= 2:
                continue
            expected[i] = value

        data = a.padding_at((0, 0, 0), 2, corners=False, edges=False)
        self.assertEqual(expected.shape, data.shape)
        self.assertEqual(str(expected), str(data))

        expected2 = expected[1:-1, 1:-1, 1:-1]
        data2 = a.padding_at((0, 0, 0), 1, corners=False, edges=False)
        self.assertEqual(expected2.shape, data2.shape)
        self.assertEqual(str(expected2), str(data2))

        expected3_tmp = np.zeros((6, 6, 6, 3), dtype=int)
        for n, i in enumerate(np.ndindex(6, 6, 6)):
            value = np.array([1, 2, 3]) + (n + 1) * 3
            expected3_tmp[i] = value
        expected3 = np.zeros((8, 8, 8, 3), dtype=int)
        expected3[1:-1, 1:-1, 1:-1] = expected3_tmp
        data3 = a.padding_at((0, 0, 0), 3, corners=False, edges=False)
        self.assertEqual(expected3.shape, data3.shape)
        self.assertEqual(str(expected3), str(data3))

    def test_padding_no_corners_edges(self):
        a = ChunkGrid(2, int, -1)
        expected = np.zeros((6, 6, 6), int)

        for n, i in enumerate(np.ndindex(6, 6, 6)):
            a.set_value(i, n)
            expected[i] = n

        self.assertEqual(expected.shape, tuple(a.size()))
        self.assertEqual(str(expected), str(a.to_dense()))

        padded1 = a.padding_at((1, 1, 1), 1, corners=True, edges=True)
        expected1 = expected[1:-1, 1:-1, 1:-1]
        self.assertEqual(str(expected1), str(padded1))

        padded2 = a.padding_at((1, 1, 1), 1, corners=False, edges=False)
        expected2 = np.full((4, 4, 4), -1, int)
        expected2[1:-1, 1:-1, :] = expected[2:-2, 2:-2, 1:-1]
        expected2[1:-1, :, 1:-1] = expected[2:-2, 1:-1, 2:-2]
        expected2[:, 1:-1, 1:-1] = expected[1:-1, 2:-2, 2:-2]
        self.assertEqual(str(expected2), str(padded2))
