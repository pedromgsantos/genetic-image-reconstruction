from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.models.triangle_solution import TriangleSolution
from src.utils import rgb_to_lab


class TriangleCIELABSolution(TriangleSolution):
    target: ClassVar[np.ndarray | None] = None
    _target_lab: ClassVar[np.ndarray | None] = None
    _target_key: ClassVar[tuple[int, tuple[int, ...], str] | None] = None

    @classmethod
    def _get_target_lab(cls) -> np.ndarray:
        if cls.target is None:
            raise ValueError("TriangleCIELABSolution.target must be set before calling fitness().")

        target_key = (id(cls.target), cls.target.shape, str(cls.target.dtype))
        if cls._target_lab is None or cls._target_key != target_key:
            cls._target_lab = rgb_to_lab(cls.target)
            cls._target_key = target_key

        return cls._target_lab

    @classmethod
    def evaluate_population_fitness(cls, population) -> None:
        pending = [individual for individual in population if individual._fitness is None]
        if not pending:
            return

        target_lab = cls._get_target_lab()
        rendered_images = [individual.render() for individual in pending]
        rendered_lab_batch = np.stack([rgb_to_lab(image) for image in rendered_images], axis=0)
        delta_e = np.sqrt(np.sum((rendered_lab_batch - target_lab[None, ...]) ** 2, axis=-1))
        fitness_values = np.mean(delta_e, axis=(1, 2))

        for individual, fitness_value in zip(pending, fitness_values.tolist(), strict=False):
            individual._fitness = float(fitness_value)

    def fitness(self) -> float:
        if self._fitness is not None:
            return self._fitness

        type(self).evaluate_population_fitness([self])
        return self._fitness

    def with_repr(self, new_repr):
        return TriangleCIELABSolution(repr=new_repr)

    def __repr__(self):
        return f"TriangleCIELABSolution(fitness={self.fitness():.4f})"
