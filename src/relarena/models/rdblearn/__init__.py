"""`rdblearn` — DFS features fed to a tabular foundation model.

Importing this package registers the `rdblearn` model. See `model`.
"""

from relarena.models.rdblearn.model import RDBLEARN_SPACE, RDBLearnModel

__all__ = ["RDBLEARN_SPACE", "RDBLearnModel"]
