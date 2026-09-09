"""Check installed estimator constructors without fitting or making API requests."""

from __future__ import annotations

import argparse
from importlib.metadata import version

import numpy as np

from tabpfn_rel.tfm import TFM_REGISTRY


def main() -> None:
    """Construct classification and regression estimators with context overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["local", "api", "all"])
    args = parser.parse_args()
    for name, dependency in (
        ("tabpfn-v3", "tabpfn"),
        ("tabpfn-v3-api", "tabpfn-client"),
    ):
        if args.backend == "local" and name.endswith("api"):
            continue
        if args.backend == "api" and not name.endswith("api"):
            continue
        spec = TFM_REGISTRY[name]
        for make in (spec.make_classifier, spec.make_regressor):
            estimator = make(
                device="cpu",
                seed=7,
                n_estimators=2,
                inference_config={
                    "SUBSAMPLE_SAMPLES": [np.array([0, 1]), np.array([1, 2])]
                },
            )
            assert estimator.random_state == 7
            assert estimator.n_estimators == 2
            if name.endswith("api"):
                assert estimator.model_path in {
                    "v3_default",
                    "tabpfn-v3-classifier-v3_default.ckpt",
                    "tabpfn-v3-regressor-v3_default.ckpt",
                }
                assert estimator.inference_config["SUBSAMPLE_SAMPLES"] == [
                    [0, 1],
                    [1, 2],
                ]
        print(
            f"{dependency} {version(dependency)}: classifier and regressor constructors passed"
        )


if __name__ == "__main__":
    main()
