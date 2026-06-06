from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image, ImageDraw

from src.config import CANVAS_W, CANVAS_H, N_TRIANGLES
from src.models.base_solution import Solution
from src.operators.triangle_genome import random_triangle


class TriangleSolution(Solution):
    target: ClassVar[np.ndarray | None] = None

    def random_initial_representation(self):
        return [random_triangle() for _ in range(N_TRIANGLES)]

    def render(self) -> np.ndarray:
        canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 255))
        for tri in self.repr:
            x1, y1, x2, y2, x3, y3, r, g, b, a = tri
            overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.polygon(
                [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3))],
                fill=(int(r), int(g), int(b), int(a * 255)),
            )
            canvas = Image.alpha_composite(canvas, overlay)
        return np.array(canvas.convert("RGB"), dtype=np.uint8)

    def save_render(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.render(), mode="RGB").save(output_path)
        return output_path

    def fitness(self) -> float:
        if self._fitness is not None:
            return self._fitness

        if TriangleSolution.target is None:
            raise ValueError("TriangleSolution.target must be set before calling fitness().")

        rendered = self.render().astype(np.float32)
        target = TriangleSolution.target.astype(np.float32)
        self._fitness = float(np.sqrt(np.mean((rendered - target) ** 2)))
        return self._fitness

    def with_repr(self, new_repr):
        return TriangleSolution(repr=new_repr)

    def __repr__(self):
        return f"TriangleSolution(fitness={self.fitness():.4f})"
