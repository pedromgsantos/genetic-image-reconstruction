from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.models.triangle_solution import TriangleSolution


class TriangleMSESolution(TriangleSolution):
    """
    Triangle solution using Mean Squared Error (MSE) as fitness metric.
    MSE is the average squared difference between pixels in the original and reconstructed images.
    """

    target: ClassVar[np.ndarray | None] = None

    def fitness(self) -> float:
        if self._fitness is not None:
            return self._fitness

        if TriangleMSESolution.target is None:
            raise ValueError("TriangleMSESolution.target must be set before calling fitness().")

        rendered = self.render().astype(np.float32)
        target = TriangleMSESolution.target.astype(np.float32)
        self._fitness = float(np.mean((rendered - target) ** 2))
        return self._fitness

    def with_repr(self, new_repr):
        return TriangleMSESolution(repr=new_repr)

    def __repr__(self):
        return f"TriangleMSESolution(fitness={self.fitness():.4f})"
