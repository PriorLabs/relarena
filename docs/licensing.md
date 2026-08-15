# Licensing

relarena is licensed under the **Apache License, Version 2.0**. `LICENSE` is the
license; `NOTICE` is the attribution that travels with it. Those two files are
authoritative — this page only says where things live.

## Scope

Apache-2.0 covers everything in this repository, including the TabPFN-Rel model
code (`src/relarena/models/tabpfn_rel/`). Moving released code elsewhere later
would not change the license under which this version was published.

Two things it does **not** cover:

- **Vendored third-party code.** The RelBench, RelGNN, and RelGT model building
  blocks are MIT and stay MIT. `NOTICE` lists them; the upstream license texts are
  in `src/relarena/models/VENDORED-LICENSES`.
- **Datasets.** None are distributed with this package. `relbench` fetches them at
  runtime, and they remain subject to their own upstream terms.

## Dependency licenses

`tabpfn` is not Apache-2.0 — it ships the Prior Labs License, which adds an
attribution requirement. It is confined to the `rdblearn` and `tabpfn-rel-*`
extras, so a plain install does not acquire it. Read the license itself for what
it obliges; the README summarizes it for anyone installing those extras.

`workflows/audit_licenses.py --check` fails if a dependency that reaches users is
copyleft, non-commercial, or declares a license it does not recognize. It runs in
CI on every change to `pyproject.toml` or `uv.lock`, and `--output` writes the
full per-package report, which CI keeps as a build artifact rather than committing.
