# Licensing

relarena is licensed under the **Apache License, Version 2.0**. `LICENSE` is the
license; `NOTICE` is the attribution that travels with it. Those two files are
authoritative — this page only says where things live.

## Scope

Apache-2.0 covers the RelArena code in this repository. The separate `tabpfn-rel`
package distributes its model code under its own Apache-2.0 `LICENSE` and `NOTICE`.

Two things it does **not** cover:

- **Vendored third-party code.** The RelBench, RelGNN, and RelGT model building
  blocks are MIT and stay MIT. `NOTICE` lists them; the upstream license texts are
  in `packages/relarena/src/relarena/models/VENDORED-LICENSES`.
- **Datasets.** None are distributed with this package. `relbench` fetches them at
  runtime, and they remain subject to their own upstream terms.

## Dependency licenses

`tabpfn` is Apache-2.0 from 9.0.0, the version `uv.lock` pins. The `rdblearn`
extra also allows 8.x, which ships the Prior Labs License, an Apache-2.0
derivative with an attribution requirement. The model weights it downloads at
runtime are licensed separately: TabPFN-2 weights
under the Prior Labs License, which adds an attribution requirement, and
TabPFN-2.5 weights under a non-commercial license. `tabpfn` is confined to the
`rdblearn` and `tabpfn-rel-*` extras, so a plain install does not acquire it.
Read the weight licenses themselves for what they oblige; the README summarizes
them for anyone installing those extras.

`workflows/audit_licenses.py --check` fails if a dependency that reaches users is
copyleft, non-commercial, or declares a license it does not recognize. It runs in
CI on every change to `pyproject.toml` or `uv.lock`, and `--output` writes the
full per-package report, which CI keeps as a build artifact rather than committing.
