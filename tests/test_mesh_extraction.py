import unittest

from reconstruction.mesh_extraction import *
from reconstruction.mesh_extraction import _detect_cut_edges


class TestDilate(unittest.TestCase):
    def test_detect_cut_edges_1(self):
        block = np.zeros((2, 2, 2), dtype=bool)
        face_segments = np.zeros((2, 2, 2, 6), dtype=bool)

        block[0, 0, 0] = True
        block[0, 1, 0] = True
        block[1, 0, 0] = True
        block[1, 1, 0] = True

        face_segments[0, 0, 0] = [0, 0, 0, 0, 1, 0]
        face_segments[0, 1, 0] = [0, 0, 0, 0, 1, 0]
        face_segments[1, 0, 0] = [0, 0, 0, 0, 1, 0]
        face_segments[1, 1, 0] = [0, 0, 0, 0, 1, 0]

        result = _detect_cut_edges(block, face_segments)

        expected = np.zeros((2, 2, 2), dtype=np.int8)
        expected[0, 0, 0] = CutEdge_X | CutEdge_Y
        expected[0, 1, 0] = CutEdge_X | CutEdge_Y
        expected[1, 0, 0] = CutEdge_X | CutEdge_Y
        expected[1, 1, 0] = CutEdge_X | CutEdge_Y

        self.assertEqual(str(expected), str(result))

    def test_detect_cut_edges_2(self):
        block = np.zeros((2, 2, 2), dtype=bool)
        face_segments = np.zeros((2, 2, 2, 6), dtype=bool)

        block[0, 0, 0] = True
        block[1, 0, 1] = True
        block[0, 1, 1] = True

        face_segments[0, 0, 0] = [1, 0, 1, 0, 0, 1]
        face_segments[1, 0, 1] = [1, 0, 1, 0, 0, 1]
        face_segments[0, 1, 1] = [1, 0, 1, 0, 0, 1]

        result = _detect_cut_edges(block, face_segments)

        expected = np.zeros((2, 2, 2), dtype=np.int8)
        expected[0, 0, 0] = CutEdge_X | CutEdge_Y
        expected[1, 0, 1] = CutEdge_Z | CutEdge_Y
        expected[0, 1, 1] = CutEdge_X | CutEdge_Z

        self.assertEqual(str(expected), str(result))

    def test_real_data_cut_edges_1(self):
        block = np.zeros((2, 2, 2), dtype=bool)
        block[0, 0, 0] = True
        block[0, 0, 1] = True
        block[1, 0, 1] = True
        face_segments = np.zeros((2, 2, 2, 6), dtype=bool)
        face_segments[0, 0, 0] = [0, 1, 0, 1, 0, 0]
        face_segments[0, 0, 1] = [0, 1, 0, 1, 1, 0]
        face_segments[1, 0, 1] = [0, 0, 0, 1, 0, 0]

        result = _detect_cut_edges(block, face_segments)

        expected = np.zeros((2, 2, 2), dtype=np.int8)
        expected[0, 0, 0] = CutEdge_NONE
        expected[0, 0, 1] = CutEdge_NONE
        expected[1, 0, 1] = CutEdge_NONE

        self.assertEqual(str(expected), str(result))

    # def test_real_data_cut_edges_2(self):
    #     block = np.zeros((2, 2, 2), dtype=bool)
    #     block[0, 0, 0] = True
    #     block[0, 1, 0] = True
    #     block[1, 0, 0] = True
    #     block[1, 1, 1] = True
    #     face_segments = np.zeros((2, 2, 2, 6), dtype=bool)
    #     face_segments[0, 0, 0] = [0, 0, 0, 0, 0, 1]
    #     face_segments[0, 1, 0] = [1, 0, 1, 0, 0, 1]
    #     face_segments[1, 0, 0] = [1, 0, 1, 0, 0, 1]
    #     face_segments[1, 1, 1] = [0, 0, 0, 0, 0, 1]
    #
    #     result = _detect_cut_edges(block, face_segments)
    #
    #     expected = np.zeros((2, 2, 2), dtype=np.int8)
    #     self.fail("TODO")
    #     expected[0, 0, 0] = CutEdge_NONE
    #     expected[0, 0, 1] = CutEdge_NONE
    #     expected[1, 0, 1] = CutEdge_NONE
    #
    #     self.assertEqual(str(expected), str(result))
