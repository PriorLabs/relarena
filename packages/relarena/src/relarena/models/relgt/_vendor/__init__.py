"""Vendored RelGT model code (snap-stanford/relgt), kept verbatim.

The relarena adapter (fit/predict, tokenization, search space) lives in
`relarena.models.relgt.model` and `relarena.models.relgt.tokenize`; this package
holds only the upstream model, imported via `RelGT`.
"""

from relarena.models.relgt._vendor.model import RelGT

__all__ = ["RelGT"]
