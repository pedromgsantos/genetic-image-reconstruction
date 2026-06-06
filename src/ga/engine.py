from copy import deepcopy
from typing import Any, Callable
import random
import numpy as np
from PIL import Image
from src.config import RESULTS_DIR


GA_DEFAULTS = {
    "xo_prob": 0.9,
    "mut_prob": 0.05,
    "elitism": True,
    "n_elites": 1,
    "verbose": False,
    "return_metrics": False,
    "diversify_on_plateau": False,
    "plateau_window": 20,
    "plateau_epsilon": 1e-3,
    "mut_sigma_start": 1.5,
    "mut_sigma_end": 0.2,
    "hc_refine_every": 0,
    "hc_iterations": 100,
    "hc_neighbors": 10,
    "hc_mut_prob": 0.05,
    "hc_sigma_scale": 0.1,
    "hc_mut_method": None,
    "run_name": None,
    "save_best_per_generation": False,
}


def _ensure_population_fitness(population):
    """Ensure all individuals in a population have up-to-date fitness values.

    Args:
        population: Sequence of solution instances.
    """
    if not population:
        return

    solution_class = type(population[0])
    if hasattr(solution_class, "evaluate_population_fitness"):
        solution_class.evaluate_population_fitness(population)
        return

    for individual in population:
        individual.fitness()

def _normalized_entropy_from_counts(counts):
    """Compute normalized Shannon entropy from count values.

    Args:
        counts: Iterable of category counts.

    Returns:
        Entropy normalized to the range [0, 1].
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1: return 0.0
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(probs.size)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0

def _build_genotype_matrix(population):
    """Build a 2D genotype matrix from flattened individual representations.

    Args:
        population: Sequence of solution instances.

    Returns:
        A matrix with shape `(n_individuals, n_genes_truncated)`.
    """
    flat_repr = [np.asarray(ind.repr, dtype=np.float64).reshape(-1) for ind in population]
    if not flat_repr: return np.empty((0, 0), dtype=np.float64)
    min_len = min(vec.size for vec in flat_repr)
    if min_len == 0: return np.empty((len(flat_repr), 0), dtype=np.float64)
    return np.vstack([vec[:min_len] for vec in flat_repr])

def _population_metrics(population):
    """Compute diversity and variance metrics for a population.

    Args:
        population: Sequence of solution instances.

    Returns:
        Dictionary containing phenotypic/genotypic entropy and variance metrics.
    """
    if not population:
        return {"phenotypic_entropy": 0.0, "genotypic_entropy": 0.0, "fitness_variance": 0.0, "genotypic_variance": 0.0}

    _ensure_population_fitness(population)
    
    fitness_values = np.array([ind.fitness() for ind in population], dtype=np.float64)
    fitness_variance = float(np.var(fitness_values))
    
    if np.allclose(fitness_values, fitness_values[0]):
        phenotypic_entropy = 0.0
    else:
        bin_count = int(np.clip(np.sqrt(len(fitness_values)), 2, 20))
        hist, _ = np.histogram(fitness_values, bins=bin_count)
        phenotypic_entropy = _normalized_entropy_from_counts(hist)
        
    genotype_matrix = _build_genotype_matrix(population)
    if genotype_matrix.size == 0:
        genotypic_variance, genotypic_entropy = 0.0, 0.0
    else:
        genotypic_variance = float(np.mean(np.var(genotype_matrix, axis=0)))
        quantized = np.round(genotype_matrix, 3)
        per_locus = [_normalized_entropy_from_counts(np.unique(quantized[:, j], return_counts=True)[1]) for j in range(quantized.shape[1])]
        genotypic_entropy = float(np.mean(per_locus)) if per_locus else 0.0
        
    return {"phenotypic_entropy": phenotypic_entropy, "genotypic_entropy": genotypic_entropy, "fitness_variance": fitness_variance, "genotypic_variance": genotypic_variance}

def get_best_individual(population):
    """Return the individual with the lowest fitness.

    Args:
        population: Sequence of solution instances.

    Returns:
        Best individual according to `fitness()` minimization.
    """
    _ensure_population_fitness(population)
    return min(population, key=lambda ind: ind.fitness())


def get_best_individuals(population, n_elites=1):
    """Return the top-N individuals with lowest fitness.

    Args:
        population: Sequence of solution instances.
        n_elites: Number of elite individuals to return.

    Returns:
        Sorted list of elite individuals.
    """
    if n_elites <= 0:
        return []
    _ensure_population_fitness(population)
    return sorted(population, key=lambda ind: ind.fitness())[:n_elites]

def diversify_population(population, percentage=0.15):
    """Replace the worst individuals with newly sampled random individuals.

    Args:
        population: Sequence of solution instances.
        percentage: Fraction of the population to replace.

    Returns:
        Diversified population list.
    """
    num_to_replace = int(len(population) * percentage)
    if num_to_replace <= 0: return population
    _ensure_population_fitness(population)
    sorted_pop = sorted(population, key=lambda ind: ind.fitness())
    template = sorted_pop[0]
    for i in range(1, num_to_replace + 1):
        sorted_pop[-i] = type(template)() # Generate completely random individual
    return sorted_pop

def hill_climbing_refine(
    individual,
    mut_method,
    mut_prob,
    sigma_scale,
    n_iterations=100,
    n_neighbors=10,
):
    """Run hill-climbing local search to refine a single individual.

    Args:
        individual: Starting solution instance.
        mut_method: Mutation callable used to generate neighbors.
        mut_prob: Mutation probability passed to `mut_method`.
        sigma_scale: Mutation scale passed to `mut_method`.
        n_iterations: Number of hill-climbing iterations.
        n_neighbors: Number of neighbors sampled per iteration.

    Returns:
        Refined individual (or original if no improving neighbor is found).
    """
    current = deepcopy(individual)
    current_fitness = current.fitness()
    for _ in range(n_iterations):
        neighbors = [mut_method(current, mut_prob, sigma_scale=sigma_scale) for _ in range(n_neighbors)]
        _ensure_population_fitness(neighbors)
        best_neighbor = get_best_individual(neighbors)
        if best_neighbor.fitness() < current_fitness:
            current = best_neighbor
            current_fitness = best_neighbor.fitness()
    return current

def genetic_algorithm(
    initial_population,
    max_generations,
    selection_algorithm: Callable,
    xo_method: Callable,
    mut_method: Callable,
    settings: dict[str, Any] | None = None,
    xo_prob: float | None = None,
    mut_prob: float | None = None,
):
    """Run a generational genetic algorithm with optional memetic refinement.

    Args:
        initial_population: Initial sequence of solution instances.
        max_generations: Number of generations to run.
        selection_algorithm: Parent selection callable.
        xo_method: Crossover callable.
        mut_method: Mutation callable.
        settings: Optional configuration dictionary overriding GA defaults.
        xo_prob: Optional crossover probability override.
        mut_prob: Optional mutation probability override.

    Returns:
        If `return_metrics` is enabled in settings, returns
        `(best_solution, fitness_history, metrics_history)`.
        Otherwise returns `(best_solution, fitness_history)`.

    Raises:
        TypeError: If `settings` contains unsupported keys.
    """
    config = dict(GA_DEFAULTS)
    unexpected_keys = set(settings or {}) - set(config)
    if unexpected_keys:
        unknown = ", ".join(sorted(unexpected_keys))
        raise TypeError(f"Unsupported genetic_algorithm setting(s): {unknown}")

    if settings:
        config.update(settings)
    if xo_prob is not None:
        config["xo_prob"] = xo_prob
    if mut_prob is not None:
        config["mut_prob"] = mut_prob

    xo_prob = config["xo_prob"]
    mut_prob = config["mut_prob"]
    verbose = config["verbose"]
    return_metrics = config["return_metrics"]

    population = deepcopy(initial_population)
    _ensure_population_fitness(population)
    best_ever = deepcopy(get_best_individual(population))
    elite_count = 0
    if config["elitism"]:
        elite_count = max(1, int(config["n_elites"]))
        elite_count = min(elite_count, len(population))
    fitness_history = [best_ever.fitness()]
    metrics_history = None
    if return_metrics:
        metrics_history = {
            "phenotypic_entropy": [],
            "genotypic_entropy": [],
            "fitness_variance": [],
            "genotypic_variance": [],
            "avg_fitness": [],
        }
        initial_metrics = _population_metrics(population)
        for key, value in initial_metrics.items():
            metrics_history[key].append(value)
        metrics_history["avg_fitness"].append(float(np.mean([ind.fitness() for ind in population])))
    
    last_diversify_gen = 0
    out_dir = None
    
    if config["save_best_per_generation"] and config["run_name"]:
        out_dir = RESULTS_DIR / config["run_name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("gen_*.png", "generation_*.png", "timelapse.gif"):
            for stale_path in out_dir.glob(pattern):
                stale_path.unlink()

    for gen in range(1, max_generations + 1):
        sigma_scale = config["mut_sigma_start"] + (config["mut_sigma_end"] - config["mut_sigma_start"]) * (gen / max_generations)
        new_population = []
        
        if elite_count:
            elites = get_best_individuals(population, elite_count)
            new_population.extend(deepcopy(elite) for elite in elites)
            
        # Standard P09 Loop
        while len(new_population) < len(population):
            first_ind = selection_algorithm(population)
            second_ind = selection_algorithm(population)
            
            if random.random() < xo_prob:
                offspr1, offspr2 = xo_method(first_ind, second_ind)
            else:
                offspr1, offspr2 = deepcopy(first_ind), deepcopy(second_ind)
                
            offspr1 = mut_method(offspr1, mut_prob, sigma_scale=sigma_scale)
            new_population.append(offspr1)
            
            if len(new_population) < len(population):
                offspr2 = mut_method(offspr2, mut_prob, sigma_scale=sigma_scale)
                new_population.append(offspr2)
                
        population = new_population
        _ensure_population_fitness(population)
        gen_best = get_best_individual(population)
        fitness_history.append(gen_best.fitness())
        
        if gen_best.fitness() < best_ever.fitness():
            best_ever = deepcopy(gen_best)

        gen_metrics = _population_metrics(population) if (return_metrics or verbose) else None
        if return_metrics:
            metrics_history["avg_fitness"].append(float(np.mean([ind.fitness() for ind in population])))
            for key, value in gen_metrics.items():
                metrics_history[key].append(value)

        if config["hc_refine_every"] > 0 and gen % config["hc_refine_every"] == 0:
            if verbose: print(f"  [HC] Refining best_ever at gen {gen} (current fitness {best_ever.fitness():.4f})")
            refined = hill_climbing_refine(
                best_ever,
                config["hc_mut_method"] or mut_method,
                mut_prob=config["hc_mut_prob"],
                sigma_scale=config["hc_sigma_scale"],
                n_iterations=config["hc_iterations"],
                n_neighbors=config["hc_neighbors"],
            )
            if refined.fitness() < best_ever.fitness():
                best_ever = refined
                if verbose: print(f"  [HC] Improved to {best_ever.fitness():.4f}")
                worst_idx = int(np.argmax([ind.fitness() for ind in population]))
                population[worst_idx] = deepcopy(refined)

        if config["diversify_on_plateau"] and gen > config["plateau_window"] and (gen - last_diversify_gen) >= config["plateau_window"] // 2:
            if np.std(fitness_history[-config["plateau_window"]:]) < config["plateau_epsilon"]:
                if verbose: print(f"  [Diversity] Plateau detected at gen {gen}, injecting random individuals...")
                population = diversify_population(population, percentage=0.15)
                last_diversify_gen = gen
                
        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"Gen {gen:>4d} | best = {best_ever.fitness():.4f} | H_p = {gen_metrics['phenotypic_entropy']:.3f} | H_g = {gen_metrics['genotypic_entropy']:.3f}")
            
        if out_dir is not None:
            img_array = best_ever.render()
            if hasattr(img_array, "astype"):
                img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(out_dir / f"gen_{gen:04d}.png")
            
    if return_metrics: return best_ever, fitness_history, metrics_history
    return best_ever, fitness_history