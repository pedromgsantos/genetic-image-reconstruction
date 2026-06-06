from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch

from src.models.triangle_solution import TriangleSolution
from src.utils import compute_embedding_cosine_distances_batch


class TriangleEmbeddingSolution(TriangleSolution):
    """
    Triangle solution scored by cosine distance in a vision embedding space.

    Default backbone is DINOv2. Lower fitness is better.
    """

    target: ClassVar[np.ndarray | None] = None
    model_name: ClassVar[str] = "facebook/dinov2-base"
    batch_size: ClassVar[int] = 32

    _device: ClassVar[torch.device | None] = None

    @classmethod
    def _default_device(cls) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def _device_changed(cls, resolved_device: torch.device) -> bool:
        if cls._device is None:
            return True
        return cls._device.type != resolved_device.type or cls._device.index != resolved_device.index

    @classmethod
    def configure_runtime(
        cls,
        *,
        device: str | torch.device | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        if model_name is not None and model_name != cls.model_name:
            cls.model_name = model_name

        resolved_device = cls._default_device() if device is None else torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            resolved_device = torch.device("cpu")

        if batch_size is not None:
            if int(batch_size) <= 0:
                raise ValueError("batch_size must be a positive integer.")
            cls.batch_size = int(batch_size)

        if cls._device_changed(resolved_device):
            cls._device = resolved_device

    @classmethod
    def runtime_summary(cls) -> str:
        cls.configure_runtime()
        return f"{cls.model_name} on {cls._device}"

    @classmethod
    def evaluate_population_fitness(cls, population) -> None:
        pending = [individual for individual in population if individual._fitness is None]
        if not pending:
            return

        if cls.target is None:
            raise ValueError("TriangleEmbeddingSolution.target must be set before calling fitness().")

        cls.configure_runtime()
        rendered_images = [individual.render() for individual in pending]
        embedding_distances = compute_embedding_cosine_distances_batch(
            rendered_images,
            cls.target,
            model_name=cls.model_name,
            device=str(cls._device),
            batch_size=cls.batch_size,
        )

        for individual, fitness_value in zip(pending, embedding_distances.tolist(), strict=False):
            individual._fitness = float(fitness_value)

    def fitness(self) -> float:
        if self._fitness is not None:
            return self._fitness

        type(self).evaluate_population_fitness([self])
        return self._fitness

    def with_repr(self, new_repr):
        return TriangleEmbeddingSolution(repr=new_repr)

    def __repr__(self):
        return f"TriangleEmbeddingSolution(fitness={self.fitness():.4f})"
