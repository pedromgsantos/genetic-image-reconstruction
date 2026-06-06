"""Mutation operators for triangle-based genomes."""

import random
from copy import deepcopy
import numpy as np
from src.config import CANVAS_H, CANVAS_W
from src.operators.triangle_genome import random_triangle

def gaussian_mutation(individual, mut_prob=0.05, sigma_coords=8.0, sigma_color=12.0, sigma_alpha=0.05, sigma_scale=1.0):
    """Mutate triangle genes by adding Gaussian noise.

    Args:
        individual: Solution instance whose `repr` will be mutated.
        mut_prob: Independent mutation probability per gene.
        sigma_coords: Base standard deviation for coordinate genes.
        sigma_color: Base standard deviation for RGB color genes.
        sigma_alpha: Base standard deviation for alpha gene.
        sigma_scale: Multiplicative scale applied to all sigma values.

    Returns:
        A new individual with mutated representation.
    """
    new_repr = deepcopy(individual.repr)
    sc = sigma_coords * sigma_scale
    scl = sigma_color * sigma_scale
    sa = sigma_alpha * sigma_scale
    
    for tri in new_repr:
        # Mutate coordinates (x, y)
        for idx in range(6):
            if random.random() < mut_prob:
                limit = CANVAS_W if idx % 2 == 0 else CANVAS_H
                tri[idx] = int(np.clip(tri[idx] + np.random.normal(0, sc), 0, limit))
        
        # Mutate colors (R, G, B)
        for idx in range(6, 9):
            if random.random() < mut_prob:
                tri[idx] = int(np.clip(tri[idx] + np.random.normal(0, scl), 0, 255))
        
        # Mutate alpha (transparency)
        if random.random() < mut_prob:
            tri[9] = float(np.clip(tri[9] + np.random.normal(0, sa), 0.0, 1.0))
            
    return individual.with_repr(new_repr)

def triangle_swap_mutation(individual, mut_prob=0.05, sigma_scale=1.0):
    """Swap triangle positions to perturb drawing order.

    Args:
        individual: Solution instance whose `repr` will be mutated.
        mut_prob: Probability of swapping each triangle position.
        sigma_scale: Unused compatibility argument for shared mutation signature.

    Returns:
        A new individual with swapped triangle ordering.
    """
    new_repr = deepcopy(individual.repr)
    n = len(new_repr)
    for i in range(n):
        if random.random() < mut_prob:
            j = random.randint(0, n - 1)
            new_repr[i], new_repr[j] = new_repr[j], new_repr[i]
    return individual.with_repr(new_repr)

def mixed_mutation(individual, mut_prob=0.05, sigma_scale=1.0, swap_weight=0.3):
    """Apply either structural swap or Gaussian mutation.

    Args:
        individual: Solution instance whose `repr` will be mutated.
        mut_prob: Mutation probability passed to the selected operator.
        sigma_scale: Scale factor passed to Gaussian mutation.
        swap_weight: Probability of choosing swap mutation.

    Returns:
        A new mutated individual.
    """
    if random.random() < swap_weight:
        return triangle_swap_mutation(individual, mut_prob=mut_prob, sigma_scale=sigma_scale)
    return gaussian_mutation(individual, mut_prob=mut_prob, sigma_scale=sigma_scale)

def random_reset_mutation(individual, mut_prob=0.01, sigma_scale=1.0):
    """Replace selected triangles with freshly sampled random triangles.

    Args:
        individual: Solution instance whose `repr` will be mutated.
        mut_prob: Probability of resetting each triangle.
        sigma_scale: Unused compatibility argument for shared mutation signature.

    Returns:
        A new individual with random-reset triangle genes.
    """
    new_repr = [
        random_triangle() if random.random() < mut_prob else deepcopy(tri)
        for tri in individual.repr
    ]
    return individual.with_repr(new_repr)