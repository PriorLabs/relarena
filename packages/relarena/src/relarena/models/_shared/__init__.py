"""Code shared between model wrappers.

Not models: the registry walks `models/*` and skips this package. The layout
encodes who shares what — a family subpackage (`gbdt`, `tfm`, `gnn`) holds code
shared *within* that family, while a module at this level is shared *across*
families. Import the submodules directly; these are internals and may move.
"""
