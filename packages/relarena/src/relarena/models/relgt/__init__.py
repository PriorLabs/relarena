"""`relgt` — the Relational Graph Transformer.

Importing this package registers the `relgt` model. See `model`; `tokenize` is the
relarena driver for the sampler, and `_vendor` holds the upstream RelGT stack.
"""

from relarena.models.relgt.model import (
    RelGTModel,
    relgt_search_space,
)

__all__ = ["RelGTModel", "relgt_search_space"]
