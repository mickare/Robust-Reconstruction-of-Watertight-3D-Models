import os
from dataclasses import dataclass
from typing import Tuple, Union, Sequence

import numpy as np
import pywavefront

from model import ModelLoader

Vec3f = Union[np.ndarray, Sequence[float], Tuple[float, float, float]]


@dataclass
class Mesh:
    """
    First simple data structure holding only the vertices and faces in a numpy array

    @param vertices th positions of triangle corners (x,y,z)
    @param faces the triangles (Triple of vertices indices)
    """
    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def from_pywavefront(cls, obj: pywavefront.Wavefront) -> "Mesh":
        """
        Load a mesh from a pywavefront object
        :param obj:
        :return:
        """
        assert obj.mesh_list
        return cls(
            vertices=np.array(obj.vertices),
            faces=np.array(obj.mesh_list[0].faces)
        )

    @classmethod
    def from_file_obj(cls, file: str, **kwargs) -> "Mesh":
        """
        Load a mesh from a .obj file
        :param file:
        :param kwargs:
        :return:
        """
        assert os.path.isfile(file), f"Mesh file is missing: {file}"
        kwargs.setdefault("encoding", "UTF-8")
        return cls.from_pywavefront(pywavefront.Wavefront(file, collect_faces=True, **kwargs))

    def get_centroids(self) -> np.ndarray:
        return self.vertices[self.faces[:, :3]].mean(axis=1)

    def scale(self, factor: float):
        """
        Scale the mesh
        :param factor:
        :return:
        """
        self.vertices *= factor
        return self

    def box(self) -> Tuple[Vec3f, Vec3f]:
        """
        Get the bounding box
        :return:
        """
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def size(self) -> Vec3f:
        """
        Get the size of the mesh
        :return:
        """
        a, b = self.box()
        return b - a

    def move(self, offset: Vec3f):
        """
        Move the mesh
        :param offset:
        :return:
        """
        self.vertices += offset

    def span_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the triangle span components of each surface with the offset v1
        :return:
            Tuple of the three triangle spans
        """
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        c = (tmp.T / np.sqrt(np.linalg.norm(tmp, axis=1))).T
        return a, b, c

    @property
    def span(self) -> np.ndarray:
        """
        Calculates the triangle spans of each surface with the offset v1.
        The span components are ordered in columns.
        :return:
            triangles Nx3x3
        """
        a, b, c = self.span_components()
        return np.transpose((a, b, c), (1, 2, 0))

    @property
    def v1(self):
        return self.vertices[self.faces[:, 0]]

    def get_dimension(self) -> int:
        return self.faces.shape[1]

    def is_fourth_dimension(self) -> bool:
        return self.get_dimension() == 4

    def to_fourth_dimension(self, copy=True) -> "Mesh":
        if self.is_fourth_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        a, b, c = self.span_components()
        v4 = self.v1 + c
        new_vertices = np.concatenate((self.vertices, v4), axis=0)
        v4_indices = np.arange(len(self.vertices), len(self.vertices) + len(c))
        new_faces = np.concatenate((self.faces, v4_indices.reshape((-1, 1))), axis=1)
        return Mesh(new_vertices, new_faces)

    def is_third_dimension(self) -> bool:
        return self.faces.shape[1] == 3

    def to_third_dimension(self, copy=True) -> "Mesh":
        if self.is_third_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        new_faces = self.faces[:, :3]
        new_vertices = self.vertices[:np.max(new_faces) + 1]
        return Mesh(new_vertices, new_faces)

    def transpose(self, shape=(0, 1, 2)):
        shape = np.asarray(shape)
        assert shape.shape == (3,)
        return Mesh(
            vertices=self.vertices[:, shape],
            faces=self.faces
        )

    def normals(self) -> np.ndarray:
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        vns = np.cross(v2 - v1, v3 - v1)
        return (vns.T / np.linalg.norm(vns, axis=1)).T


class MeshModelLoader(ModelLoader):

    def __init__(self, number_of_samples: int = 10000, noise: float = 0.0):
        self.number_of_samples = number_of_samples
        self.noise = noise

    def sample_random_from(self, mesh: Mesh, number: int) -> np.ndarray:
        # Calculate probability of each face
        v1, v2, v3 = mesh.vertices[mesh.faces.T]
        a = v2 - v1
        b = v3 - v1
        areas = np.cross(a, b)
        prob = np.linalg.norm(areas, axis=1)
        prob /= prob.sum()
        # Sample random faces
        face_samples = np.random.choice(len(mesh.faces), number, p=prob)
        # Sample random point on each sampled face
        v1, v2, v3 = mesh.vertices[mesh.faces[face_samples].T]
        u = np.random.random(number)
        v = np.random.random(number) * (1.0 - u)
        a = v2 - v1
        b = v3 - v1
        return v1 + (u * a.T).T + (v * b.T).T

    def load(self, path: str, **kwargs) -> np.ndarray:
        assert path.endswith(".obj")
        mesh = Mesh.from_file_obj(path, **kwargs)
        return self.sample_random_from(mesh, self.number_of_samples)
