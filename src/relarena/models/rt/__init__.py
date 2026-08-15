"""`rt-plurel` — the Relational Transformer, fine-tuned per task from RT-P.

Importing this package registers the `rt-plurel` model. Every configured value
lives in `config`; see `model` for the wrapper and `export` for the tensor
export.
"""

from relarena.models.rt.model import RT_SPACE, RTPluRelModel, clear_scratch

__all__ = ["RT_SPACE", "RTPluRelModel", "clear_scratch"]
