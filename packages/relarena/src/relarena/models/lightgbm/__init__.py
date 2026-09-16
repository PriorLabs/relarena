"""`lightgbm` — gradient-boosted trees on entity-only features.

Importing this package registers the `lightgbm` model. See `model`.
"""

from relarena.models.lightgbm.model import LIGHTGBM_SPACE, LightGBMModel

__all__ = ["LIGHTGBM_SPACE", "LightGBMModel"]
