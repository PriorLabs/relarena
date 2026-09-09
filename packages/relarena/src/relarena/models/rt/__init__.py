"""`rt-plurel` — the Relational Transformer, fine-tuned per task from RT-P.

Importing this package registers the `rt-plurel` system. Every configured value
lives in `config`; see `model` for the wrapper and `export` for the tensor
export.
"""

from relarena.models.rt.model import RTPluRelSystem, clear_scratch

__all__ = ["RTPluRelSystem", "clear_scratch"]
