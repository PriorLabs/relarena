"""Register the complete staged KurveRSC system for benchmark and user data.

The optional KurveRSC package owns search, selection and final fitting. This
adapter passes the caller's censored splits unchanged and returns predictions;
the harness alone owns hidden-label scoring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from relbench.base import EntityTask, TaskType

from relarena_core import InnerSplit, OuterSplit, RelArenaSystem, register_system
from relarena_core.registry import registry


@register_system
class KurveRSC20261022System(RelArenaSystem):
    """The versioned, complete staged KurveRSC prediction procedure."""

    name = "kurversc-2026-10-22"
    supported_task_types = frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )
    complete_pipeline_output: Path | None = None

    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
    ) -> np.ndarray:
        """Run the packaged recipe on the supplied, target-unseen protocol."""
        if seed != 0:
            raise ValueError(
                "The complete staged KurveRSC recipe uses deployment seed 0; "
                "other seeds are not silently substituted."
            )
        try:
            from kurversc.protocol_runner import run_protocol_task
        except ImportError as exc:
            raise ImportError(
                "Install the candidate relarena[kurversc] extra: the complete "
                "procedure requires kurversc[staged]>=0.2.0."
            ) from exc

        identity = self.run_identity
        result = run_protocol_task(
            task,
            inner_split=inner_split,
            outer_split=outer_split,
            dataset_name=identity.dataset if identity else "database",
            task_name=identity.task if identity and identity.task else "prediction",
            seed=seed,
            time_limit=time_limit,
            cache_directory=self.cache.directory,
            run_identity=identity,
        )
        self.complete_pipeline_output = result.output
        return result.predictions["prediction"].to_numpy(dtype=float)


registry.register_alias("kurversc-latest", KurveRSC20261022System.name)

__all__ = ["KurveRSC20261022System"]
