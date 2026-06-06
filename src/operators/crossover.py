"""Crossover operators for triangle-based genomes."""

import random
from copy import deepcopy


def _blend_triangle(triangle1, triangle2, weight):
    """Blend two triangle genes with a scalar weight.

    Args:
        triangle1: First triangle gene sequence.
        triangle2: Second triangle gene sequence.
        weight: Blend factor in [0, 1].

    Returns:
        A blended triangle gene list.
    """
    blended = []
    for idx, (value1, value2) in enumerate(zip(triangle1, triangle2)):
        mixed = weight * value1 + (1.0 - weight) * value2
        if idx < 9:
            blended.append(int(round(mixed)))
        else:
            blended.append(float(mixed))
    return blended

def single_point_crossover(parent1, parent2):
    """Perform single-point crossover over triangle lists.

    Args:
        parent1: First parent individual.
        parent2: Second parent individual.

    Returns:
        Tuple of two offspring individuals.
    """
    point = random.randint(1, len(parent1.repr) - 1)
    repr1 = parent1.repr[:point] + parent2.repr[point:]
    repr2 = parent2.repr[:point] + parent1.repr[point:]
    return parent1.with_repr(deepcopy(repr1)), parent2.with_repr(deepcopy(repr2))

def uniform_crossover(parent1, parent2):
    """Perform uniform crossover at triangle granularity.

    Args:
        parent1: First parent individual.
        parent2: Second parent individual.

    Returns:
        Tuple of two offspring individuals.
    """
    repr1, repr2 = [], []
    for t1, t2 in zip(parent1.repr, parent2.repr):
        if random.random() < 0.5:
            repr1.append(deepcopy(t1))
            repr2.append(deepcopy(t2))
        else:
            repr1.append(deepcopy(t2))
            repr2.append(deepcopy(t1))
    return parent1.with_repr(repr1), parent2.with_repr(repr2)


def arithmetic_crossover(parent1, parent2):
    """Blend matching triangle genes using arithmetic recombination.

    Args:
        parent1: First parent individual.
        parent2: Second parent individual.

    Returns:
        Tuple of two offspring individuals.
    """
    repr1, repr2 = [], []
    for tri1, tri2 in zip(parent1.repr, parent2.repr):
        weight = random.random()
        repr1.append(_blend_triangle(tri1, tri2, weight))
        repr2.append(_blend_triangle(tri2, tri1, weight))
    return parent1.with_repr(repr1), parent2.with_repr(repr2)


def order_based_crossover(parent1, parent2):
    """Perform order-based crossover preserving relative gene order.

    Args:
        parent1: First parent individual.
        parent2: Second parent individual.

    Returns:
        Tuple of two offspring individuals.
    """
    n_genes = len(parent1.repr)
    if n_genes < 2:
        return deepcopy(parent1), deepcopy(parent2)

    keep_count = random.randint(1, n_genes - 1)
    keep_indices = set(random.sample(range(n_genes), keep_count))

    child1 = [None] * n_genes
    child2 = [None] * n_genes

    for idx in keep_indices:
        child1[idx] = deepcopy(parent1.repr[idx])
        child2[idx] = deepcopy(parent2.repr[idx])

    fill1 = [deepcopy(gene) for idx, gene in enumerate(parent2.repr) if idx not in keep_indices]
    fill2 = [deepcopy(gene) for idx, gene in enumerate(parent1.repr) if idx not in keep_indices]

    fill1_iter = iter(fill1)
    fill2_iter = iter(fill2)
    for idx in range(n_genes):
        if child1[idx] is None:
            child1[idx] = next(fill1_iter)
        if child2[idx] is None:
            child2[idx] = next(fill2_iter)

    return parent1.with_repr(child1), parent2.with_repr(child2)