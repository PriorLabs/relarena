"""Declarative hyperparameter search spaces, decoupled from the model.

This follows AutoGluon / TabArena's protocol: a model class is just
`fit` / `predict` + identity, and its hyperparameter search space is a
**separate object** bound to the model in the registry — rather than methods on
the model class. A space is one of:

  * a ConfigSpace `ConfigurationSpace` (sampled randomly), or
  * an explicit ordered `fixed_grid` of configs (enumerated — for small discrete
    spaces like a DFS depth; order is the trial priority, since the harness caps the
    grid to its budget, so put the most important config, e.g. the default, first),

plus a single `default_overrides` config (the "zero-tuning" regime reported as the
default). It is prepended for a `space` so it always runs; for a `fixed_grid` it
runs only if it's an entry within the budget cap — so order grids default-first.

The model author writes the space; the **harness** owns the budget (how many
configs to draw) — that split is the fairness lever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ConfigSpace import ConfigurationSpace

logger = logging.getLogger(__name__)


def _to_python(config: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy scalars in a sampled config to plain Python types."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in config.items()}


@dataclass
class SearchSpace:
    """A model's hyperparameter search space (see module docstring).

    Provide at most one of `space` / `fixed_grid` (the two are mutually
    exclusive). With neither, the model is untunable and only its
    `default_overrides` config runs (e.g. a parameter-free baseline).
    """

    #: The single "default"/zero-tuning config, given as overrides on the model's
    #: underlying library defaults. `{}` means *no overrides* — i.e. pass no
    #: config and run the library's own defaults. Tagged and reported as the
    #: "default" regime; for a `fixed_grid` it should be one of the grid entries
    #: (a warning is logged otherwise), ordered early so it survives the budget cap.
    default_overrides: dict[str, Any]
    #: A continuous/large space **sampled randomly** under the harness's trial
    #: budget. Mutually exclusive with `fixed_grid`.
    space: ConfigurationSpace | None = None
    #: A small *discrete* set of configs **enumerated in order**. The harness caps it
    #: to the trial budget (keeps the first `n_trials`), so order = priority: put the
    #: most important configs (e.g. the `default_overrides`) first.
    #: Mutually exclusive with `space`.
    fixed_grid: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Validate the `space`/`fixed_grid` choice and the default config.

        `space` and `fixed_grid` are mutually exclusive. For a `fixed_grid`,
        `default_overrides` should be one of the grid entries — otherwise the
        default regime is never run or tagged, so we warn (rather than fail, since
        a missing default is recoverable and shouldn't block a run).
        """
        if self.space is not None and self.fixed_grid is not None:
            raise ValueError(
                "A SearchSpace has either `space` or `fixed_grid`, not both."
            )
        if (
            self.fixed_grid is not None
            and self.default_overrides not in self.fixed_grid
        ):
            logger.warning(
                "default_overrides %s is not in the fixed_grid; the default regime "
                "will not be run or tagged for this search space.",
                self.default_overrides,
            )

    @property
    def is_tunable(self) -> bool:
        """Whether there is anything to tune beyond the default config."""
        return self.space is not None or self.fixed_grid is not None

    def configs(self, n_trials: int, seed: int) -> list[dict[str, Any]]:
        """Return the ordered configs to evaluate (before tagging).

        * `fixed_grid` -> the grid as given, capped at `n_trials` (warns if it
          drops entries — so put the most important configs first);
        * `space` -> the `default_overrides` config plus `n_trials` configs
          sampled from the space (seeded by `seed` for reproducibility);
        * neither -> just the `default_overrides` config.
        """
        if self.fixed_grid is not None:
            configs = list(self.fixed_grid)
            if len(configs) > n_trials:
                logger.warning(
                    "Fixed grid has %d configs but the trial budget is "
                    "n_trials=%d; evaluating only the first %d and dropping %d.",
                    len(configs),
                    n_trials,
                    n_trials,
                    len(configs) - n_trials,
                )
                configs = configs[:n_trials]
            return configs

        configs = [dict(self.default_overrides)]
        if self.space is not None and n_trials > 0:
            self.space.seed(seed)
            sampled = self.space.sample_configuration(n_trials)
            if n_trials == 1:  # ConfigSpace returns a single config, not a list
                sampled = [sampled]
            configs += [_to_python(dict(c)) for c in sampled]
        return configs


@dataclass(frozen=True)
class TaskStats:
    """Cheap dataset statistics available at tuning time, for size-aware spaces.

    Passed to a search-space *factory* (see `SearchSpaceProvider`) so a model
    can pick its grid from the task's scale — e.g. a coarser grid on large datasets,
    mirroring the compute-budget choices made for such methods in the literature.
    """

    #: Number of seed entities in the inner-split train table (the "training nodes").
    num_train_nodes: int


#: What a model registers as its search space: either a fixed `SearchSpace`
#: (the common case) or a factory that builds one from `TaskStats` (opt-in,
#: for spaces that legitimately depend on dataset scale). The harness resolves a
#: factory once per task, when the stats are known (see `resolve_search_space`).
SearchSpaceProvider = SearchSpace | Callable[[TaskStats], SearchSpace]


def resolve_search_space(
    provider: SearchSpaceProvider, stats: TaskStats
) -> SearchSpace:
    """Return the concrete `SearchSpace` for `stats`.

    Calls `provider` if it is a factory; returns it unchanged if it is already a
    `SearchSpace` (which is not callable), so static spaces are a no-op.
    """
    return provider(stats) if callable(provider) else provider
