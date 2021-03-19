from typing import Tuple, Sequence

import maxflow
import numpy as np
import tqdm

from reconstruction.data.chunks import ChunkGrid
from reconstruction.data.faces import ChunkFace
from reconstruction.mathlib import Vec3i

MinCutNodeIndex = Tuple[Vec3i, ChunkFace]


# =====================================================================
# MinCut
# =====================================================================


class MinCut:
    """
    Creates a MinCut through a voxel crust by connecting an outer and inner surface via octahedral subgraphs per voxel.
    """
    _segments = None
    _grid_segment0 = None
    _grid_segment1 = None

    # Static const variables
    SEGMENT_OUTER_ID = 0  # Source
    SEGMENT_INNER_ID = 1  # Sink

    def __init__(self, diff: ChunkGrid[float], crust: ChunkGrid[bool], crust_outer: ChunkGrid[bool],
                 crust_inner: ChunkGrid[bool], s=4, a=1e-20):
        self.crust = crust

        # Method cache
        get_node = MinCut.get_node

        self.weights = (diff ** s) + a
        self.weights[~crust] = -1
        self.weights.cleanup(remove=True)

        self.voxels = {tuple(p): w for p, w in self.weights.items(mask=crust) if w >= 0}
        self.nodes = list(set(get_node(p, f) for p in self.voxels.keys() for f in ChunkFace))
        self.nodes_index = {f: n for n, f in enumerate(self.nodes)}
        nodes_index = self.nodes_index

        nodes_count = len(self.nodes)

        self.graph = maxflow.Graph[float](nodes_count, nodes_count)
        self.graph.add_nodes(nodes_count)

        # Method cache
        grap_add_edge = self.graph.add_edge
        grap_add_tedge = self.graph.add_tedge
        nodes_index_get = self.nodes_index.get
        # Name cache
        CF = ChunkFace

        # visited: Set[Tuple[Tuple[Vec3i, CF], Tuple[Vec3i, CF]]] = set()
        for vPos, w in tqdm.tqdm(self.voxels.items(), total=len(self.voxels), desc="Linking Faces"):
            iN = nodes_index[get_node(vPos, CF.NORTH)]
            iS = nodes_index[get_node(vPos, CF.SOUTH)]
            iT = nodes_index[get_node(vPos, CF.TOP)]
            iB = nodes_index[get_node(vPos, CF.BOTTOM)]
            iE = nodes_index[get_node(vPos, CF.EAST)]
            iW = nodes_index[get_node(vPos, CF.WEST)]
            for f, o in [
                (iN, iE), (iN, iW), (iN, iT), (iN, iB),
                (iS, iE), (iS, iW), (iS, iT), (iS, iB),
                (iT, iE), (iT, iW), (iB, iE), (iB, iW)
            ]:  # type: ChunkFace
                grap_add_edge(f, o, w, w)

        # Source
        for vPos in tqdm.tqdm(list(crust_outer.where()), desc="Linking Source"):
            for f in ChunkFace:  # type: ChunkFace
                fNode = get_node(tuple(vPos), f)
                fIndex = nodes_index_get(fNode, None)
                if fIndex is not None:
                    grap_add_tedge(fIndex, 10000, 0)

        # Sink
        for vPos in tqdm.tqdm(list(crust_inner.where()), desc="Linking Sink"):
            for f in ChunkFace:  # type: ChunkFace
                fNode = get_node(tuple(vPos), f)
                fIndex = nodes_index_get(fNode, None)
                if fIndex is not None:
                    grap_add_tedge(fIndex, 0, 10000)

    def segments(self):
        if self._segments is None:
            flow = self.graph.maxflow()
            self._segments = np.asanyarray(self.graph.get_grid_segments(np.arange(len(self.nodes))))
        return self._segments

    def grid_segments(self) -> Tuple[ChunkGrid[np.bool8], ChunkGrid[np.bool8]]:
        to_voxel = MinCut.to_voxel
        segments = self.segments()
        if self._grid_segment0 is None:
            self._grid_segment0 = ChunkGrid(self.crust.chunk_size, np.bool8, False)
            self._grid_segment0[[p for node, s in zip(self.nodes, segments) if s == False
                                 for p in to_voxel(node)]] = True
        if self._grid_segment1 is None:
            self._grid_segment1 = ChunkGrid(self.crust.chunk_size, np.bool8, False)
            self._grid_segment1[[p for node, s in zip(self.nodes, segments) if s == True
                                 for p in to_voxel(node)]] = True
        return self._grid_segment0, self._grid_segment1

    @staticmethod
    def get_node(pos: Vec3i, face: ChunkFace) -> MinCutNodeIndex:
        """Basically forces to have only positive-direction faces"""
        if face % 2 == 0:
            return pos, face
        else:
            return tuple(np.add(pos, face.direction(), dtype=int)), face.flip()

    @staticmethod
    def to_voxel(nodeIndex: MinCutNodeIndex) -> Sequence[Vec3i]:
        pos, face = nodeIndex
        return [
            pos,
            np.asarray(face.direction()) * (face.flip() % 2) + pos
        ]
