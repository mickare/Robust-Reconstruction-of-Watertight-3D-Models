"""
Example configurations for the models that worked well
"""
import enum
from typing import Dict

import numpy as np

from model.bunny import FixedBunny
from model.dragon import Dragon
from model.model_mesh import MeshModelLoader
from model.model_pts import PtsModelLoader


class Example(enum.Enum):
    BunnyFixed = 0
    Bunny = 1
    Dragon = 2
    Cat = 3
    Dog = 4
    Camel = 5


example_config: Dict[Example, Dict] = {
    Example.BunnyFixed: dict(
        dilations_max=5,
        dilations_reverse=1
    ),
    Example.Bunny: dict(
        dilations_max=30,
        dilations_reverse=3
    ),
    Example.Dragon: dict(
        dilations_max=20,
        dilations_reverse=3
    ),
    Example.Cat: dict(
        dilations_max=20,
        dilations_reverse=4
    ),
    Example.Dog: dict(
        dilations_max=20,
        dilations_reverse=4
    ),
    Example.Camel: dict(
        dilations_max=20,
        dilations_reverse=3
    )
}


def example_load(example: Example) -> np.ndarray:
    if example == Example.BunnyFixed:
        return FixedBunny.bunny()
    if example == Example.Bunny:
        return PtsModelLoader().load("models/bunny/bunnyData.pts")
    if example == Example.Dragon:
        return Dragon().load()
    if example == Example.Cat:
        return MeshModelLoader(samples=30000, noise=0.01).load("models/cat/cat_reference.obj")
    if example == Example.Dog:
        return MeshModelLoader(samples=30000, noise=0.01).load("models/dog/dog_reference.obj")
    if example == Example.Camel:
        return MeshModelLoader(samples=60000, noise=0.01).load("models/camel-poses/camel-reference.obj")
