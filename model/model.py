from abc import ABC, abstractmethod

import numpy as np


class ModelLoader(ABC):

    @abstractmethod
    def load(self, path: str) -> np.ndarray:
        ...
