# Project guidance

## Purpose and structure

This repository models statistical learning of tone sequences on a two-community network.  The core model learns directed associations with an exponentially decaying Hebbian trace, turns association strength into two-alternative forced-choice predictions, and compares several post-exposure consolidation/replay mechanisms.  The notebooks use simulation, behavioural-data fitting, and bootstrap analyses to evaluate the model.

- `src/model.py`: model parameters, learning, readout, and veridical, generative, and selective replay consolidation.
- `src/experiment.py`: fixed experimental graph, random walks, and balanced trial-table generation.
- `src/plotNet.py` and `src/plotBehav.py`: reusable network and behavioural plotting functions.
- `*.ipynb`: exploratory/reproducible analyses. `Hebbian_exponential_decay.ipynb` runs simulations and consolidation comparisons; `fit_behav_beta.ipynb` estimates the trace-decay parameter against behavioural data; `matrix_bootstrap_analysis.ipynb` bootstraps community-separation measures.
- Root `.pkl` and `.svg` files are saved analysis outputs. Do not overwrite them unless an analysis is intentionally being rerun.

## Python conventions already used here

- Use Python with `numpy` for numerical state and vectorised calculations, `pandas` DataFrames for trial/analysis tables, and `matplotlib`/`seaborn`/`networkx` for visualisation.
- Keep reusable domain logic in `src/`; keep analysis orchestration, fitting, data loading, and figure export in notebooks.
- Import project modules explicitly, e.g. `from src.model import HebbianSequenceModel`. Keep third-party imports at the top of each module/notebook section.
- Represent node/sequence identifiers as integer `numpy` arrays. Convert public method inputs with `np.asarray(..., dtype=int)` or `int(...)` where the current API does so.
- Treat the adjacency/weight matrix convention as directed: row = cue/source and column = target. Exclude self-transitions before normalisation or probability calculations.
- Pass a `numpy.random.Generator` (`rng`) into functions that sample. Use a fixed seed in simulations/notebooks when results must be reproducible; do not use global random state.
- Return rich tables from model phases: preserve the input trial DataFrame and add computed columns rather than changing its layout. Existing standard columns include `block`, `trial_in_block`, `cue`, `target`, `sequence`, `legal`, `within`, `trial_type`, and `regularity`.
- Use `@dataclass` parameter containers with scientifically meaningful defaults. Extend `ModelParameters` or `ConsolidationParameters` rather than scattering new constants through simulation code.
- Use `ValueError` for invalid user-facing model inputs and `NotImplementedError` for intentionally unavailable mechanisms.

## Naming and formatting

- Follow the repository's current naming: `snake_case` for functions, methods, variables, and DataFrame columns; `PascalCase` for classes; `UPPER_CASE` only for genuine constants.
- Prefer clear, domain-specific names (`trace_decay`, `replay_length`, `familiarity_normalized`) over abbreviations. Existing loop indices such as `iTrial`, `iBlc`, and `iSubj` are legacy notebook style; use readable `snake_case` names in new code when practical.
- Match the surrounding file's formatting: four-space indentation, spaces around binary operators and keyword arguments, and one logical operation per line when calls become long. Use single quotes in new Python source unless the surrounding code uses double quotes for DataFrame keys.
- Add short comments for scientific/mechanistic decisions (e.g. normalisation, replay order, or readout timing), not for obvious syntax. Use concise docstrings for public functions and non-obvious algorithms.

## Safe change and verification practice

- Preserve the meanings of `beta`, learning rates, retention, and readout modes; changes to these parameters alter the scientific model and should be explicit.
- Keep the public trial-table schema and the output fields of `transition_readout()`/`run_test_phase()` backwards compatible unless all dependent notebooks are updated together.
- For core-model changes, at minimum run a small seeded end-to-end check: build the graph, generate trials, learn an exposure sequence, run the test phase, and exercise the relevant consolidation mode.
- There is no dedicated automated test suite or dependency manifest in the repository. Do not assume a formatting, linting, or test command exists; report the exact validation performed.
