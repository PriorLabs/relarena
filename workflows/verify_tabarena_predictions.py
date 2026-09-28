"""Manually check exported predictions against TabArena's result reader.

Run after changing the prediction artifact format; CI does not run this check.
Use a Python environment with TabArena, AutoGluon core and NumPy installed.
These optional verification dependencies are not installed by this script.

From the repository root, run:
    python workflows/verify_tabarena_predictions.py /path/to/artifacts

The verify(directory) entrypoint reads completed results.pkl files recursively
and checks metrics and simulation inputs. Only use trusted artifact directories:
loading pickle files can execute code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from autogluon.core.metrics import get_metric
from tabarena.benchmark.result import BaselineResult, ConfigResult


def verify(directory: Path) -> None:
    """Check result loading, metric errors and simulation inputs without conversion."""
    paths = sorted(directory.rglob("results.pkl"))
    if not paths:
        raise ValueError(f"No completed result artifacts in {directory}")
    for path in paths:
        result = BaselineResult.from_pickle(str(path))
        assert isinstance(result, ConfigResult), path
        metric = get_metric(result.result["metric"], problem_type=result.problem_type)
        for split in ("val", "test"):
            labels = getattr(result, f"y_{split}")
            predictions = getattr(result, f"y_pred_proba_{split}")
            expected = result.result[
                "metric_error_val" if split == "val" else "metric_error"
            ]
            np.testing.assert_allclose(metric.error(labels, predictions), expected)
        simulation = result.generate_old_sim_artifact()[result.dataset][
            result.split_idx
        ]
        np.testing.assert_array_equal(
            simulation["pred_proba_dict_test"][result.framework],
            result.y_pred_proba_test,
        )
        assert (
            result.hyperparameters["hyperparameters"]
            == result.result["method_metadata"]["model_hyperparameters"]
        )
        assert len(result.compute_df_result()) == 1
        print(f"Verified {path}")
    print(f"Verified {len(paths)} completed config artifacts with TabArena")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
