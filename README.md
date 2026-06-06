# Galapagos — Triangle Genetic Algorithm

Project for the Computational Intelligence for Optimization course, Master's in Data Science and Advanced Analytics, NOVA IMS.

This repository reflects a cleaned version of the project, with improved organization and documentation for clarity and reproducibility.

---

## Project Overview

*Girl with a Pearl Earring* (Vermeer, 1665) is approximated by composing 100 semi-transparent
RGBA triangles on a 300×400 canvas. A genetic algorithm evolves the triangle positions, sizes,
and colors over successive generations to minimize the visual difference between the composed
image and the original.

<p align="center">
  <img src="https://github.com/pedromgsantos/genetic-image-reconstruction/blob/main/data/girl_pearl_earing.png" width="200" alt="Target image"/>
  <img src="https://github.com/pedromgsantos/genetic-image-reconstruction/blob/main/results/tuned_rmse/timelapse.gif" width="200" alt="Evolutionary reconstruction timelapse"/>
  <br/>
  <sub>Target image (left) — Evolutionary reconstruction timelapse, RMSE (right)</sub>
</p>

The project explores four fitness metrics — pixel error, perceptual color distance, and
deep visual embeddings — and uses random search over operator combinations to identify the
best-performing configuration for each metric.

---

## Project Goals

1. **GA Engine** – Implement a generational genetic algorithm with elitism, plateau
   diversification, hill climbing, and a linear mutation sigma schedule.
2. **Operator Library** – Build interchangeable selection, crossover, and mutation operators
   so that configurations can be swapped and benchmarked without changing the core loop.
3. **Fitness Metric Comparison** – Compare four fitness functions (RMSE, MSE, CIELAB Delta E,
   DINOv2 cosine distance) across identical operator configurations.
4. **Hyperparameter Search** – Run a random search over operator and hyperparameter combinations,
   persisting all results to CSV for reproducible analysis.
5. **Analysis and Reporting** – Statistically compare metric and operator performance across
   seeds and produce timelapse GIFs of the evolutionary process.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── girl_pearl_earing.png        # Target image
├── notebooks/                        # Main workflow (run in order)
│   ├── 01_experiment_tracking.ipynb
│   ├── 02_rmse_tuning.ipynb
│   └── 03_random_search_analysis.ipynb
├── results/                          # Run artifacts (not tracked in Git)
│   ├── random_search_history.csv
│   ├── rmse/
│   ├── mse/
│   ├── cielab_delta_e/
│   ├── embedding_cosine/
│   └── tuned_rmse/
└── src/
    ├── config.py                     # Global constants and paths
    ├── utils.py                      # Experiment execution, search, plots, GIFs
    ├── analysis_utils.py             # Statistical analysis for random search history
    ├── ga/
    │   └── engine.py                 # Generational GA loop
    ├── models/                       # Solution representations and fitness functions
    │   ├── base_solution.py
    │   ├── triangle_solution.py      # RMSE
    │   ├── triangle_mse_solution.py  # MSE
    │   ├── triangle_cielab_solution.py  # CIELAB Delta E
    │   └── triangle_embedding_solution.py  # DINOv2 cosine distance
    └── operators/                    # Pluggable GA operators
        ├── selection.py              # fitness_proportionate, ranking, tournament
        ├── crossover.py              # single_point, uniform, arithmetic, order_based
        ├── mutation.py               # gaussian, triangle_swap, mixed, random_reset
        └── triangle_genome.py        # Genome sampling
```

---

## Notebooks

- **01_experiment_tracking.ipynb** – loads the target image, runs random search per metric,
  and executes the full experiments across all four fitness functions
- **02_rmse_tuning.ipynb** – RMSE-focused tuning workflow using the best configuration
  recovered from the random search history
- **03_random_search_analysis.ipynb** – comparative statistical analysis of operator and
  hyperparameter performance across all runs saved in `results/random_search_history.csv`

---

## Core Modules (`src/`)

- **config.py** – canvas dimensions (`300×400`), triangle count (`100`), default seed,
  and project paths
- **ga/engine.py** – generational GA loop with optional elitism, plateau diversification,
  hill climbing, and population diversity metrics
- **models/** – one solution class per fitness metric; each encodes an individual as a list
  of 100 triangles `[x1, y1, x2, y2, x3, y3, r, g, b, a]` and implements its own fitness call
- **operators/** – interchangeable selection, crossover, and mutation implementations;
  all operators are stateless and can be composed freely
- **utils.py** – random search driver, `run_and_report_experiment`, auxiliary metrics,
  matplotlib plots, and GIF timelapse generation
- **analysis_utils.py** – statistical comparisons and ranking utilities for the
  saved random search CSV history

---

## Fitness Metrics

| Class | Metric | Notes |
|---|---|---|
| `TriangleSolution` | RGB RMSE | Default metric for tuning |
| `TriangleMSESolution` | RGB MSE | Penalizes large errors more aggressively |
| `TriangleCIELABSolution` | Mean Delta E | Perceptually uniform color distance |
| `TriangleEmbeddingSolution` | DINOv2 cosine distance | Semantic similarity via `facebook/dinov2-base` |

---

## How to Run

Typical execution order:

1. `01_experiment_tracking.ipynb` – run random search and full experiments per metric
2. `02_rmse_tuning.ipynb` – deeper tuning on the best RMSE configuration
3. `03_random_search_analysis.ipynb` – statistical analysis of all saved results

The project assumes execution from the repository root. Notebooks already adjust `sys.path`
accordingly. There is no CLI entry point — use the notebooks or call helper functions from
`src/utils.py` directly.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Technical Notes

- `TriangleEmbeddingSolution` selects `cuda` automatically when available, otherwise falls
  back to `cpu`. The first run downloads `facebook/dinov2-base` from HuggingFace.
- Random search defaults: `num_experiments = 48`, `max_gens = 200`, `num_seeds = 30`,
  `pop_size = 100` (fixed during search).
- `load_best_configs_from_history()` reconstructs the best operator configuration from
  `results/random_search_history.csv` without re-running experiments.
- Run artifacts (fitness CSVs, timelapse GIFs) are saved under `results/<run_name>/`
  and are not tracked in Git.
