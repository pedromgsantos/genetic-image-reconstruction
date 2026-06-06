"""Triangle genome encoding and sampling utilities."""

import random

from src.config import CANVAS_H, CANVAS_W


def encode_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a):
    """Encode triangle geometry and color into a flat gene list.

    Args:
        x1: X coordinate of vertex 1.
        y1: Y coordinate of vertex 1.
        x2: X coordinate of vertex 2.
        y2: Y coordinate of vertex 2.
        x3: X coordinate of vertex 3.
        y3: Y coordinate of vertex 3.
        r: Red channel in [0, 255].
        g: Green channel in [0, 255].
        b: Blue channel in [0, 255].
        a: Alpha value in [0.0, 1.0].

    Returns:
        Triangle gene list `[x1, y1, x2, y2, x3, y3, r, g, b, a]`.
    """
    return [x1, y1, x2, y2, x3, y3, r, g, b, a]


def random_triangle():
    """Sample a random triangle gene within canvas and color bounds.

    Returns:
        A randomly generated triangle gene list.
    """
    vertices = [
        random.randint(0, CANVAS_W),
        random.randint(0, CANVAS_H),
        random.randint(0, CANVAS_W),
        random.randint(0, CANVAS_H),
        random.randint(0, CANVAS_W),
        random.randint(0, CANVAS_H),
    ]
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    alpha = round(random.uniform(0.0, 1.0), 3)
    return encode_triangle(*vertices, *color, alpha)
