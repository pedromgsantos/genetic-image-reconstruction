"""Selection operators for genetic algorithms."""

import random
from copy import deepcopy

import numpy as np


def fitness_proportionate_selection(population):
    """Select one individual using fitness-proportionate probabilities.

    Lower fitness values are better, so values are adjusted to create
    positive selection weights.

    Args:
        population: Sequence of candidate individuals.

    Returns:
        A deep-copied selected individual.
    """
    fitness_values = [ind.fitness() for ind in population]
    worst = max(fitness_values)
    epsilon = 1e-9
    adjusted_fitness = [
        worst - f + epsilon
        for f in fitness_values
    ]

    total_fitness = sum(adjusted_fitness)
    random_nr = random.uniform(0, total_fitness)
    sliding_value = 0.0

    for ind_idx, ind in enumerate(population):
        sliding_value += adjusted_fitness[ind_idx]
        if random_nr <= sliding_value:
            return deepcopy(ind)

    return deepcopy(population[-1])


def tournament_selection(population, tournament_size=2):
    """Select one individual by tournament among random contenders.

    Args:
        population: Sequence of candidate individuals.
        tournament_size: Number of contenders sampled per tournament.

    Returns:
        A deep-copied tournament winner.
    """
    contenders = random.choices(population, k=tournament_size)
    winner = min(contenders, key=lambda ind: ind.fitness())
    return deepcopy(winner)


def ranking_selection(population):
    """Select one individual using rank-based sampling.

    Args:
        population: Sequence of candidate individuals.

    Returns:
        A deep-copied selected individual.
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness(), reverse=True)

    n = len(sorted_pop)
    weights = np.arange(1, n + 1, dtype=np.float64)
    probs = weights / weights.sum()

    selected_idx = int(np.random.choice(n, size=1, replace=True, p=probs)[0])
    return deepcopy(sorted_pop[selected_idx])
