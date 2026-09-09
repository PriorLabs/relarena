"""Check public estimator imports and the models that use them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from relarena_core import registry, tfm
from relarena_core.predict_contract import predict_to_contract


def main() -> None:
    """Verify imports in this process, then run the focused regression suite."""
    forbidden = ("relarena.models", "tabpfn", "tabpfn_client", "fastdfs")
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    )
    assert not loaded, f"Public estimator imports loaded model packages: {loaded}"
    assert not registry.names(), registry.names()
    assert callable(predict_to_contract)
    assert callable(tfm.fit_tfm) and callable(tfm.predict_tfm)
    print(f"Public estimator imports passed: {tfm.__file__}", flush=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "packages/relarena-core/tests/test_tfm.py",
            "packages/relarena-core/tests/test_predict_contract.py",
            "packages/relarena/tests/models/rdblearn",
            "packages/relarena/tests/models/lightgbm",
            "packages/relarena/tests/models/_shared/gbdt",
            "packages/relarena/tests/featurization",
            "packages/tabpfn-rel/tests",
            "packages/relarena/tests/test_refit.py",
            "packages/relarena/tests/models/dummy",
            "packages/relarena/tests/models/test_discovery.py",
            "packages/relarena/tests/userdb",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


if __name__ == "__main__":
    main()
