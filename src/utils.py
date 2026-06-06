from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import random
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from src.config import DEFAULT_SEED, RANDOM_SEARCH_NUM_SEEDS, RESULTS_DIR
from src.ga.engine import genetic_algorithm, hill_climbing_refine
from src.operators import (
    arithmetic_crossover,
    fitness_proportionate_selection,
    gaussian_mutation,
    mixed_mutation,
    order_based_crossover,
    random_reset_mutation,
    ranking_selection,
    single_point_crossover,
    tournament_selection,
    triangle_swap_mutation,
    uniform_crossover,
)


SELECTION_REGISTRY = {
    "fitness_proportionate": fitness_proportionate_selection,
    "ranking": ranking_selection,
    "tournament": tournament_selection,
}

CROSSOVER_REGISTRY = {
    "arithmetic": arithmetic_crossover,
    "order_based": order_based_crossover,
    "single_point": single_point_crossover,
    "uniform": uniform_crossover,
}

MUTATION_REGISTRY = {
    "gaussian": gaussian_mutation,
    "mixed": mixed_mutation,
    "random_reset": random_reset_mutation,
    "triangle_swap": triangle_swap_mutation,
}

POP_SIZE_SEARCH_CHOICES = (100,)
RANDOM_SEARCH_HISTORY_CSV = RESULTS_DIR / "random_search_history.csv"
RANDOM_SEARCH_HISTORY_FIELDS = (
    "solution",
    "config_id",
    "seed",
    "generation",
    "best_fitness",
    "selection",
    "crossover",
    "mutation",
    "n_elites",
    "tournament_size",
    "swap_weight",
    "xo_prob",
    "mut_prob",
    "sigma_start",
    "sigma_end",
    "plateau_window",
)

_EMBEDDING_RUNTIME_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


# ---------------------------------------------------------------------------
# Search and GA configuration helpers
# ---------------------------------------------------------------------------


def build_ga_operators(config):
    """Build configured GA operator callables from a config mapping.

    Args:
        config: Dictionary with operator names and optional operator parameters.

    Returns:
        A tuple with `(selection_algorithm, crossover_method, mutation_method)`.
    """
    selection_name = config["selection_name"]
    crossover_name = config["crossover_name"]
    mutation_name = config["mutation_name"]

    selection_algorithm = SELECTION_REGISTRY[selection_name]
    if selection_name == "tournament":
        selection_algorithm = partial(selection_algorithm, tournament_size=config["tournament_size"])

    mutation_method = MUTATION_REGISTRY[mutation_name]
    if mutation_name == "mixed":
        mutation_method = partial(mutation_method, swap_weight=config["swap_weight"])

    return selection_algorithm, CROSSOVER_REGISTRY[crossover_name], mutation_method


def _build_search_history_rows(results):
    """Convert search results into generation-level CSV rows.

    Args:
        results: List of per-configuration search result dictionaries.

    Returns:
        A list of dictionaries ready to be written into the history CSV.
    """
    rows = []
    ordered_results = sorted(results, key=lambda item: item["Experiment"])

    for result in ordered_results:
        for seed_run in result["seed_runs"]:
            best_fitness_history = np.minimum.accumulate(
                np.asarray(seed_run["generation_best_fitness"], dtype=np.float64)
            )

            for generation, best_fitness in enumerate(best_fitness_history):
                rows.append(
                    {
                        "solution": result["solution_class"],
                        "config_id": int(result["Experiment"]),
                        "seed": int(seed_run["seed"]),
                        "generation": int(generation),
                        "best_fitness": float(best_fitness),
                        "selection": result["selection_name"],
                        "crossover": result["crossover_name"],
                        "mutation": result["mutation_name"],
                        "n_elites": int(result["n_elites"]),
                        "tournament_size": result["tournament_size"],
                        "swap_weight": result["swap_weight"],
                        "xo_prob": float(result["xo_prob"]),
                        "mut_prob": float(result["mut_prob"]),
                        "sigma_start": float(result["sigma_start"]),
                        "sigma_end": float(result["sigma_end"]),
                        "plateau_window": int(result["plateau_window"]),
                    }
                )

    return rows


def _append_search_history_rows(rows, csv_path=None):
    """Append generated history rows to a CSV file.

    Args:
        rows: CSV rows produced by `_build_search_history_rows`.
        csv_path: Optional destination path. Uses default history path when omitted.

    Returns:
        The resolved CSV path used for writing.
    """
    resolved_csv_path = Path(csv_path) if csv_path is not None else RANDOM_SEARCH_HISTORY_CSV
    resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return resolved_csv_path

    needs_header = not resolved_csv_path.exists() or resolved_csv_path.stat().st_size == 0
    with resolved_csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANDOM_SEARCH_HISTORY_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)

    return resolved_csv_path


def _save_simple_fitness_history_csv(run_dir, fitness_history):
    """Save generation-level best fitness history for one run.

    Args:
        run_dir: Directory where run artifacts are stored.
        fitness_history: Sequence of best-fitness values per generation.

    Returns:
        Path to the generated CSV file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "fitness_history.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "best_fitness"])
        for generation, best_fitness in enumerate(fitness_history):
            writer.writerow([int(generation), float(best_fitness)])

    return csv_path


def _evaluate_search_config(task):
    """Evaluate one random-search configuration across multiple seeds.

    Args:
        task: Dictionary containing solution class, config, target, and run settings.

    Returns:
        A dictionary with aggregated fitness statistics and per-seed histories.
    """
    solution_class = task["solution_class"]
    solution_class_name = task["solution_class_name"]
    config = task["config"]
    pop_size = int(task["pop_size"])
    max_gens = task["max_gens"]
    target = task["target"]
    num_seeds = int(task["num_seeds"])

    if target is not None:
        solution_class.target = target

    seed_fitnesses: list[float] = []
    seed_runs = []
    for seed_idx in range(num_seeds):
        run_seed = config["ga_seed"] + seed_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        initial_pop = [solution_class() for _ in range(pop_size)]

        random.seed(run_seed)
        np.random.seed(run_seed)
        selection_algorithm, xo_method, mut_method = build_ga_operators(config)
        best_sol = genetic_algorithm(
            initial_population=initial_pop,
            max_generations=max_gens,
            selection_algorithm=selection_algorithm,
            xo_method=xo_method,
            mut_method=mut_method,
            settings=config["ga_settings"],
        )
        best_sol, fitness_history = best_sol
        final_fitness = float(best_sol.fitness())
        seed_fitnesses.append(final_fitness)
        seed_runs.append(
            {
                "seed_index": seed_idx,
                "seed": run_seed,
                "generation_best_fitness": [float(value) for value in fitness_history],
                "final_fitness": final_fitness,
            }
        )

    avg_fitness = float(np.mean(seed_fitnesses))
    std_fitness = float(np.std(seed_fitnesses))

    return {
        "solution_class": solution_class_name,
        "Experiment": config["Experiment"],
        "config_seed": config["ga_seed"],
        "pop_size": pop_size,
        "selection_name": config["selection_name"],
        "crossover_name": config["crossover_name"],
        "mutation_name": config["mutation_name"],
        "n_elites": config["ga_settings"]["n_elites"],
        "tournament_size": config["tournament_size"],
        "swap_weight": config["swap_weight"],
        "xo_prob": config["ga_settings"]["xo_prob"],
        "mut_prob": config["ga_settings"]["mut_prob"],
        "sigma_start": config["ga_settings"]["mut_sigma_start"],
        "sigma_end": config["ga_settings"]["mut_sigma_end"],
        "plateau_window": config["ga_settings"]["plateau_window"],
        "fitness": avg_fitness,
        "fitness_std": std_fitness,
        "num_seeds": num_seeds,
        "max_gens": max_gens,
        "seed_runs": seed_runs,
    }


def _run_search(
    solution_class,
    configs,
    max_gens,
    num_seeds,
    history_csv_path=None,
):
    """Execute random-search experiments and persist generation history.

    Args:
        solution_class: Solution class used to instantiate population individuals.
        configs: List of sampled GA configurations.
        max_gens: Number of generations per configuration evaluation.
        num_seeds: Number of seeds used to evaluate each configuration.
        history_csv_path: Optional output CSV path for generation-level history rows.

    Returns:
        A dictionary representing the selected best configuration.
    """
    total = len(configs)
    worker_count = min(total, os.cpu_count() or 1)
    should_parallelize = total > 1 and worker_count > 1
    mode = f"parallel ({worker_count} workers)" if should_parallelize else "sequential"
    print(f"Starting search: {total} experiments | {max_gens} generations each | {num_seeds} seeds/config | {mode}")

    target = getattr(solution_class, "target", None)
    solution_class_name = solution_class.__name__
    tasks = [
        {
            "solution_class": solution_class,
            "solution_class_name": solution_class_name,
            "config": config,
            "pop_size": config["pop_size"],
            "max_gens": max_gens,
            "target": target,
            "num_seeds": num_seeds,
        }
        for config in configs
    ]

    results = []

    if should_parallelize:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_config = {}
            for task in tasks:
                config = task["config"]
                print(
                    f"[{config['Experiment']:>2}/{total}] Starting "
                    f"pop={config['pop_size']} | "
                    f"{config['selection_name']} / {config['crossover_name']} / {config['mutation_name']} | "
                    f"seeds={num_seeds}"
                )
                future = executor.submit(_evaluate_search_config, task)
                future_to_config[future] = config
            for future in as_completed(future_to_config):
                result = future.result()
                results.append(result)
                print(
                    f"[{result['Experiment']:>2}/{total}] Finished -> "
                    f"avg fitness {result['fitness']:.4f} (std {result['fitness_std']:.4f})"
                )
    else:
        for task in tasks:
            config = task["config"]
            print(
                f"[{config['Experiment']:>2}/{total}] Starting "
                f"pop={config['pop_size']} | "
                f"{config['selection_name']} / {config['crossover_name']} / {config['mutation_name']} | "
                f"seeds={num_seeds}"
            )
            result = _evaluate_search_config(task)
            results.append(result)
            print(
                f"[{result['Experiment']:>2}/{total}] Finished -> "
                f"avg fitness {result['fitness']:.4f} (std {result['fitness_std']:.4f})"
            )

    results.sort(key=lambda result: result["fitness"])
    history_rows = _build_search_history_rows(results)
    resolved_history_csv_path = _append_search_history_rows(history_rows, csv_path=history_csv_path)

    print("\n===========================================")
    print("TOP 5 CONFIGURATIONS FOUND:")
    print("===========================================")
    for index, result in enumerate(results[:5], start=1):
        line = (
            f"{index}. fitness={result['fitness']:.4f} | "
            f"std={result['fitness_std']:.4f} | "
            f"pop={result['pop_size']} | "
            f"{result['selection_name']} / {result['crossover_name']} / {result['mutation_name']} | "
            f"elites={result['n_elites']} | "
            f"xo={result['xo_prob']} | mut={result['mut_prob']} | "
            f"sigma={result['sigma_start']}->{result['sigma_end']} | "
            f"plateau={result['plateau_window']}"
        )
        if result["tournament_size"] is not None:
            line += f" | tourn={result['tournament_size']}"
        if result["swap_weight"] is not None:
            line += f" | swap={result['swap_weight']}"
        print(line)

    print(f"Search history CSV updated: {resolved_history_csv_path}")

    best_row = results[0]
    return {
        "pop_size": int(best_row["pop_size"]),
        "selection_name": best_row["selection_name"],
        "crossover_name": best_row["crossover_name"],
        "mutation_name": best_row["mutation_name"],
        "tournament_size": best_row["tournament_size"],
        "swap_weight": best_row["swap_weight"],
        "ga_settings": {
            "xo_prob": float(best_row["xo_prob"]),
            "mut_prob": float(best_row["mut_prob"]),
            "elitism": int(best_row["n_elites"]) > 0,
            "n_elites": int(best_row["n_elites"]),
            "verbose": False,
            "diversify_on_plateau": True,
            "plateau_window": int(best_row["plateau_window"]),
            "plateau_epsilon": 0.05,
            "mut_sigma_start": float(best_row["sigma_start"]),
            "mut_sigma_end": float(best_row["sigma_end"]),
            "hc_refine_every": 0,
        },
        "search_history_csv": str(resolved_history_csv_path),
        "selected_experiment": int(best_row["Experiment"]),
        "solution_class": best_row["solution_class"],
    }


def random_search(
    solution_class,
    num_experiments=48,
    max_gens=200,
    num_seeds=RANDOM_SEARCH_NUM_SEEDS,
    history_csv_path=None,
):
    """Run random search over GA configurations for a solution class.

    Args:
        solution_class: The solution class to optimize
        num_experiments: Number of random configurations to test
        max_gens: Maximum generations for each run
        num_seeds: Number of seeds to evaluate each configuration
        history_csv_path: Optional CSV path used to append generation-level search history

    Returns:
        Dict with best hyperparameters: {
            "pop_size": int,
            "selection_name": str,
            "crossover_name": str,
            "mutation_name": str,
            "tournament_size": int | None,
            "swap_weight": float | None,
            "ga_settings": dict,
            "search_history_csv": str,
            "selected_experiment": int,
            "solution_class": str,
        }
    """
    search_rng = random.Random(DEFAULT_SEED + 1)

    configs = []
    for exp in range(num_experiments):
        sampled_pop_size = int(search_rng.choice(POP_SIZE_SEARCH_CHOICES))
        n_elites = search_rng.choice([0, 1, 2, 3])
        xo_prob = round(search_rng.uniform(0.70, 0.95), 2)
        mut_prob = round(search_rng.uniform(0.01, 0.10), 3)
        sigma_start = round(search_rng.uniform(1.0, 2.5), 2)
        sigma_end = round(search_rng.uniform(0.05, 0.3), 2)
        p_window = search_rng.choice([15, 20, 30, 40])
        selection_name = search_rng.choice(list(SELECTION_REGISTRY))
        crossover_name = search_rng.choice(list(CROSSOVER_REGISTRY))
        mutation_name = search_rng.choice(list(MUTATION_REGISTRY))
        t_size = search_rng.choice([2, 3, 4, 5]) if selection_name == "tournament" else None
        swap_w = round(search_rng.uniform(0.10, 0.50), 2) if mutation_name == "mixed" else None
        configs.append(
            {
                "Experiment": exp + 1,
                "ga_seed": DEFAULT_SEED + 2000 + exp,
                "pop_size": sampled_pop_size,
                "selection_name": selection_name,
                "crossover_name": crossover_name,
                "mutation_name": mutation_name,
                "tournament_size": t_size,
                "swap_weight": swap_w,
                "ga_settings": {
                    "xo_prob": xo_prob,
                    "mut_prob": mut_prob,
                    "elitism": n_elites > 0,
                    "n_elites": n_elites,
                    "verbose": False,
                    "diversify_on_plateau": True,
                    "plateau_window": p_window,
                    "plateau_epsilon": 0.05,
                    "mut_sigma_start": sigma_start,
                    "mut_sigma_end": sigma_end,
                    "hc_refine_every": 0,
                },
            }
        )

    return _run_search(
        solution_class=solution_class,
        configs=configs,
        max_gens=max_gens,
        num_seeds=num_seeds,
        history_csv_path=history_csv_path,
    )


def _build_ga_params_from_history_row(meta_row, pop_size, search_history_csv):
    """Reconstruct a GA parameter dictionary from one history CSV row.

    Args:
        meta_row: CSV row containing selected operator and GA hyperparameters.
        pop_size: Population size to inject into the output parameter dictionary.
        search_history_csv: Path to the history CSV used for provenance metadata.

    Returns:
        A GA parameter dictionary compatible with existing training helpers.
    """
    tournament_size = int(meta_row["tournament_size"]) if meta_row["tournament_size"] else None
    swap_weight = float(meta_row["swap_weight"]) if meta_row["swap_weight"] else None
    n_elites = int(meta_row["n_elites"])

    return {
        "pop_size": int(pop_size),
        "selection_name": meta_row["selection"],
        "crossover_name": meta_row["crossover"],
        "mutation_name": meta_row["mutation"],
        "tournament_size": tournament_size,
        "swap_weight": swap_weight,
        "ga_settings": {
            "xo_prob": float(meta_row["xo_prob"]),
            "mut_prob": float(meta_row["mut_prob"]),
            "elitism": n_elites > 0,
            "n_elites": n_elites,
            "verbose": False,
            "diversify_on_plateau": True,
            "plateau_window": int(meta_row["plateau_window"]),
            "plateau_epsilon": 0.05,
            "mut_sigma_start": float(meta_row["sigma_start"]),
            "mut_sigma_end": float(meta_row["sigma_end"]),
            "hc_refine_every": 0,
        },
        "search_history_csv": str(search_history_csv),
        "selected_experiment": int(meta_row["config_id"]),
        "solution_class": meta_row["solution"],
    }


def load_best_configs_from_history(
    csv_path=None,
    *,
    solution_names=("TriangleSolution", "TriangleMSESolution", "TriangleCIELABSolution"),
):
    """Load best GA configurations per solution from history CSV.

    The selected configuration for each solution is the one with the lowest
    mean seed-best fitness across recorded seeds.

    Args:
        csv_path: Optional path to a random-search history CSV file.
        solution_names: Iterable of solution class names to extract.

    Returns:
        {
            "search_history_csv": str,
            "best_config_by_solution": {
                solution_name: {
                    "config_id": int,
                    "mean_seed_best_fitness": float,
                    "std_seed_best_fitness": float,
                    "num_seeds": int,
                }
            },
            "params_by_solution": {
                solution_name: ga_params_dict,
            }
        }

    Raises:
        FileNotFoundError: If the history CSV does not exist.
        ValueError: If CSV contents are missing or inconsistent.
    """
    search_history_csv = Path(csv_path) if csv_path is not None else RANDOM_SEARCH_HISTORY_CSV

    if not search_history_csv.exists():
        raise FileNotFoundError(f"CSV not found: {search_history_csv}")

    with search_history_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV is empty: {search_history_csv}")

    if len(POP_SIZE_SEARCH_CHOICES) != 1:
        raise ValueError(
            "CSV does not store pop_size and POP_SIZE_SEARCH_CHOICES has multiple values. "
            "Cannot reconstruct params unambiguously."
        )
    pop_size = int(POP_SIZE_SEARCH_CHOICES[0])

    filtered_rows = [row for row in rows if row["solution"] in solution_names]
    if not filtered_rows:
        raise ValueError(
            f"No rows found in {search_history_csv} for requested solutions: {solution_names}"
        )

    seed_best = {}
    config_meta = {}

    for row in filtered_rows:
        key = (row["solution"], int(row["config_id"]), int(row["seed"]))
        fitness = float(row["best_fitness"])

        previous = seed_best.get(key)
        if previous is None or fitness < previous:
            seed_best[key] = fitness

        config_key = (row["solution"], int(row["config_id"]))
        config_meta[config_key] = row

    config_scores = {}
    for (solution_name, config_id, _seed), best in seed_best.items():
        score_key = (solution_name, config_id)
        if score_key not in config_scores:
            config_scores[score_key] = []
        config_scores[score_key].append(best)

    best_config_by_solution = {}
    for (solution_name, config_id), values in config_scores.items():
        mean_best = float(np.mean(values))
        std_best = float(np.std(values))

        current = best_config_by_solution.get(solution_name)
        if current is None or mean_best < current["mean_seed_best_fitness"]:
            best_config_by_solution[solution_name] = {
                "config_id": int(config_id),
                "mean_seed_best_fitness": mean_best,
                "std_seed_best_fitness": std_best,
                "num_seeds": len(values),
                "meta": config_meta[(solution_name, config_id)],
            }

    missing_solutions = [name for name in solution_names if name not in best_config_by_solution]
    if missing_solutions:
        raise ValueError(
            f"Missing solutions in CSV {search_history_csv}: {missing_solutions}"
        )

    params_by_solution = {}
    summary_by_solution = {}
    for solution_name, selected in best_config_by_solution.items():
        params_by_solution[solution_name] = _build_ga_params_from_history_row(
            selected["meta"],
            pop_size,
            search_history_csv,
        )
        summary_by_solution[solution_name] = {
            "config_id": selected["config_id"],
            "mean_seed_best_fitness": selected["mean_seed_best_fitness"],
            "std_seed_best_fitness": selected["std_seed_best_fitness"],
            "num_seeds": selected["num_seeds"],
        }

    return {
        "search_history_csv": str(search_history_csv),
        "best_config_by_solution": summary_by_solution,
        "params_by_solution": params_by_solution,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to CIELAB.

    Args:
        image: RGB image as a `uint8` NumPy array.

    Returns:
        CIELAB image as a floating-point NumPy array.
    """
    rgb = image.astype(np.float32) / 255.0
    rgb = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = rgb @ xyz_matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

    delta = 6 / 29
    f_xyz = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3 * delta**2) + 4 / 29,
    )

    lab = np.empty_like(f_xyz)
    lab[..., 0] = 116 * f_xyz[..., 1] - 16
    lab[..., 1] = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    lab[..., 2] = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    return lab


def compute_reconstruction_metrics(pred_image: np.ndarray, target_image: np.ndarray) -> dict[str, float]:
    """Compute reconstruction metrics between two RGB images.

    Args:
        pred_image: Predicted image as an RGB `uint8` array.
        target_image: Ground-truth image as an RGB `uint8` array.

    Returns:
        Dictionary with `rmse`, `mse`, and `delta_e` values.
    """
    pred = pred_image.astype(np.float32)
    target = target_image.astype(np.float32)
    diff = pred - target

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    pred_lab = rgb_to_lab(pred_image)
    target_lab = rgb_to_lab(target_image)
    delta_e = float(np.mean(np.sqrt(np.sum((pred_lab - target_lab) ** 2, axis=-1))))

    return {
        "rmse": rmse,
        "mse": mse,
        "delta_e": delta_e,
    }


def compute_embedding_cosine_distances_batch(
    pred_images: list[np.ndarray],
    target_image: np.ndarray,
    *,
    model_name: str = "facebook/dinov2-base",
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute embedding cosine distances for a batch against one target image.

    Args:
        pred_images: List of predicted RGB images.
        target_image: Target RGB image.
        model_name: Hugging Face model identifier for embedding extraction.
        device: Torch device string. Auto-selects CUDA when available.
        batch_size: Batch size for embedding inference.

    Returns:
        A NumPy array of cosine distances (`1 - cosine_similarity`) per image.

    Raises:
        ValueError: If `batch_size` is not a positive integer.
    """
    resolved_device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    cache_key = (model_name, resolved_device)
    if cache_key not in _EMBEDDING_RUNTIME_CACHE:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(resolved_device).eval()
        _EMBEDDING_RUNTIME_CACHE[cache_key] = (processor, model)

    processor, model = _EMBEDDING_RUNTIME_CACHE[cache_key]

    target_pil = Image.fromarray(target_image).convert("RGB")
    target_inputs = processor(images=[target_pil], return_tensors="pt")
    target_inputs = {key: value.to(resolved_device) for key, value in target_inputs.items()}

    with torch.inference_mode():
        target_outputs = model(**target_inputs)
        target_embedding = getattr(target_outputs, "pooler_output", None)
        if target_embedding is None:
            target_embedding = target_outputs.last_hidden_state[:, 0, :]
        target_embedding = F.normalize(target_embedding, dim=-1)

        distances: list[np.ndarray] = []
        pil_images = [Image.fromarray(img).convert("RGB") for img in pred_images]
        for start in range(0, len(pil_images), batch_size):
            batch = pil_images[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}

            outputs = model(**inputs)
            embedding = getattr(outputs, "pooler_output", None)
            if embedding is None:
                embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(embedding, dim=-1)

            similarities = torch.clamp(torch.matmul(embedding, target_embedding.T).squeeze(-1), -1.0, 1.0)
            distances.append((1.0 - similarities).detach().cpu().numpy())

    return np.concatenate(distances, axis=0)


# ---------------------------------------------------------------------------
# Plotting and output helpers
# ---------------------------------------------------------------------------


def run_solution_experiment(
    solution_class,
    params,
    run_name,
    fitness_label,
    hill_climbing=True,
    max_generations=6000,
):
    """Run one full GA experiment and persist artifacts.

    Args:
        solution_class: Solution class used to instantiate the population.
        params: GA parameter dictionary with operators and GA settings.
        run_name: Relative folder name under `RESULTS_DIR` for artifacts.
        fitness_label: Label used in progress logs.
        hill_climbing: Enables or disables all hill-climbing steps.
        max_generations: Number of generations for the GA run.

    Returns:
        A tuple `(best_solution, fitness_history, metrics_history)`.
    """
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    use_hill_climbing = bool(hill_climbing)
    hc_refine_every = 300
    hc_iterations = 80
    hc_neighbors = 15
    hc_mut_prob = 0.03
    hc_sigma_scale = 0.10
    final_refine_iterations = 1000
    final_refine_neighbors = 20
    final_refine_mut_prob = 0.02
    final_refine_sigma_scale = 0.05

    selection_algorithm, xo_method, mut_method = build_ga_operators(params)
    settings = {
        **params["ga_settings"],
        "verbose": True,
        "return_metrics": True,
        "run_name": run_name,
        "save_best_per_generation": True,
        "hc_refine_every": hc_refine_every if use_hill_climbing else 0,
        "hc_iterations": hc_iterations if use_hill_climbing else 0,
        "hc_neighbors": hc_neighbors if use_hill_climbing else 0,
        "hc_mut_prob": hc_mut_prob if use_hill_climbing else 0.0,
        "hc_sigma_scale": hc_sigma_scale if use_hill_climbing else 0.0,
        "hc_mut_method": gaussian_mutation if use_hill_climbing else None,
    }

    print(
        f"Starting {fitness_label} GA run with pop_size={params['pop_size']} "
        f"for {max_generations} generations..."
    )
    best_solution, fitness_history, metrics_history = genetic_algorithm(
        initial_population=[solution_class() for _ in range(params["pop_size"])],
        max_generations=max_generations,
        selection_algorithm=selection_algorithm,
        xo_method=xo_method,
        mut_method=mut_method,
        settings=settings,
    )

    if use_hill_climbing:
        refined = hill_climbing_refine(
            best_solution,
            gaussian_mutation,
            mut_prob=final_refine_mut_prob,
            sigma_scale=final_refine_sigma_scale,
            n_iterations=final_refine_iterations,
            n_neighbors=final_refine_neighbors,
        )
        if refined.fitness() < best_solution.fitness():
            best_solution = refined

    print(f"Best {fitness_label}: {best_solution.fitness():.4f}")

    best_repr_path = run_dir / "best_repr.pkl"
    with open(best_repr_path, "wb") as handle:
        pickle.dump(best_solution.repr, handle)

    best_image_path = best_solution.save_render(run_dir / "best.png")
    print(f"Saved best image: {best_image_path}")

    frame_count = len(list(run_dir.glob("gen_*.png")))
    print(f"Saved frames: {frame_count}")
    history_csv_path = _save_simple_fitness_history_csv(run_dir, fitness_history)
    print(f"Saved fitness history: {history_csv_path}")

    return best_solution, fitness_history, metrics_history


def run_and_report_experiment(
    solution_class,
    params,
    run_name,
    fitness_label,
    hill_climbing=True,
    max_generations=6000,
    *,
    seed=DEFAULT_SEED,
    target_image=None,
):
    """Run one experiment, generate GIF, and plot summary.

    Args:
        solution_class: Solution class used to instantiate the population.
        params: GA parameter dictionary with operators and GA settings.
        run_name: Relative folder name under `RESULTS_DIR` for artifacts.
        fitness_label: Label used for logging and plotting.
        hill_climbing: Enables or disables all hill-climbing steps.
        max_generations: Number of generations for the GA run.
        seed: Random seed used for Python and NumPy RNG.
        target_image: Optional RGB target image array for plotting.

    Returns:
        A tuple `(best_solution, fitness_history, metrics_history, gif_path)`.
    """
    random.seed(seed)
    np.random.seed(seed)

    best_solution, fitness_history, metrics_history = run_solution_experiment(
        solution_class=solution_class,
        params=params,
        run_name=run_name,
        fitness_label=fitness_label,
        hill_climbing=hill_climbing,
        max_generations=max_generations,
    )

    gif_path = create_run_gif(run_name)
    resolved_target = target_image if target_image is not None else solution_class.target
    plot_run_summary(
        best_solution,
        fitness_history,
        metrics_history,
        fitness_label,
        resolved_target,
    )

    print(f"Run folder for {fitness_label}: {run_name}")
    print(f"GIF: {gif_path}")

    return best_solution, fitness_history, metrics_history, gif_path


def plot_run_summary(best_solution, fitness_history, metrics_history, fitness_label, target_image):
    """Plot fitness, entropy, and reconstruction summary for one run.

    Args:
        best_solution: Best solution object returned by the GA.
        fitness_history: Sequence of best fitness values per generation.
        metrics_history: Dictionary containing entropy histories.
        fitness_label: Label used in chart titles and axis names.
        target_image: RGB target image array used for visual comparison.
    """
    gens = np.arange(len(fitness_history))
    best_rendered = best_solution.render()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(gens, fitness_history, color="#2563eb", linewidth=1.8)
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel(f"Best {fitness_label}")
    axes[0, 0].set_title(f"GA Convergence ({fitness_label})")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        gens,
        metrics_history["phenotypic_entropy"],
        color="#059669",
        linewidth=1.6,
        label="Phenotypic entropy",
    )
    axes[0, 1].plot(
        gens,
        metrics_history["genotypic_entropy"],
        color="#d97706",
        linewidth=1.6,
        label="Genotypic entropy",
    )
    axes[0, 1].set_xlabel("Generation")
    axes[0, 1].set_ylabel("Normalized entropy")
    axes[0, 1].set_title("Population Entropy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].imshow(target_image)
    axes[1, 0].set_title("Target")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(best_rendered)
    axes[1, 1].set_title(f"Best Reconstruction | {fitness_label}={best_solution.fitness():.2f}")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def create_run_gif(
    run_name,
    results_dir: str | Path = RESULTS_DIR,
    output_name="timelapse.gif",
    frame_pattern="gen_*.png",
    duration=80,
    loop=0,
):
    """Create a GIF from saved generation frames for a run.

    Args:
        run_name: Run directory name under `results_dir`.
        results_dir: Root directory where run folders are stored.
        output_name: GIF file name to create.
        frame_pattern: Glob pattern for frame discovery.
        duration: Frame duration in milliseconds.
        loop: GIF loop count (`0` means infinite loop).

    Returns:
        Path to the generated GIF file.

    Raises:
        FileNotFoundError: If no frame files match `frame_pattern`.
    """
    run_dir = Path(results_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(run_dir.glob(frame_pattern))

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {run_dir} matching {frame_pattern}.")

    frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
    output_path = run_dir / output_name

    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            format="GIF",
        )
    finally:
        for frame in frames:
            frame.close()

    print(f"GIF saved to {output_path}")
    return output_path
