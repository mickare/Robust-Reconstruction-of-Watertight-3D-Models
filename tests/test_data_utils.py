import unittest

from reconstruction.data.data_utils import ValueIter, PositionIter


class TestValueIter(unittest.TestCase):
    def test_construct_1(self):
        a = ValueIter(slice(None), -3, 5)
        self.assertEqual(-3, a.low)
        self.assertEqual(5, a.high)
        self.assertEqual(-3, a.start)
        self.assertEqual(5, a.stop)
        self.assertEqual(1, a.step)

    def test_construct_2(self):
        a = ValueIter(slice(1, -1, None), -3, 5)
        self.assertEqual(-3, a.low)
        self.assertEqual(5, a.high)
        self.assertEqual(1, a.start)
        self.assertEqual(-1, a.stop)
        self.assertEqual(1, a.step)

    def test_construct_3(self):
        a = ValueIter(slice(-4, 6, None), -3, 5)
        self.assertEqual(-3, a.low)
        self.assertEqual(5, a.high)
        self.assertEqual(-3, a.start)
        self.assertEqual(5, a.stop)
        self.assertEqual(1, a.step)

    def test_divfloor_1(self):
        a = ValueIter(slice(None), 0, 2)
        b = a // 2
        self.assertEqual(0, b.low)
        self.assertEqual(1, b.high)
        self.assertEqual(0, b.start)
        self.assertEqual(1, b.stop)
        self.assertEqual(1, b.step)

    def test_divfloor_2(self):
        a = ValueIter(slice(None, None, 3), 0, 1)
        b = a // 2
        self.assertEqual(0, b.low)
        self.assertEqual(1, b.high)
        self.assertEqual(0, b.start)
        self.assertEqual(1, b.stop)
        self.assertEqual(1, b.step)

    def test_divfloor_3(self):
        a = ValueIter(slice(None, None, 2), -3, 5)
        b = a // 2
        self.assertEqual(-2, b.low)
        self.assertEqual(3, b.high)
        self.assertEqual(-2, b.start)
        self.assertEqual(3, b.stop)
        self.assertEqual(1, b.step)

    def test_iter(self):
        a = ValueIter(slice(1, 5, 1), -3, 5)
        self.assertListEqual(list(range(1, 5)), list(a))


class TestPositionIter(unittest.TestCase):
    def test_iter(self):
        a = PositionIter(slice(None), slice(0, 1), slice(1, 2), (-2, -2, -2), (3, 3, 3))

        self.assertListEqual([
            (-2, 0, 1),
            (-1, 0, 1),
            (0, 0, 1),
            (1, 0, 1),
            (2, 0, 1)
        ], list(a))
