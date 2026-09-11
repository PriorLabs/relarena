# RelArena

RelArena benchmarks relational prediction models with shared temporal validation,
tuning and reporting. It includes baseline models and adapters for RelBench tasks.

```bash
pip install relarena
relarena --list
```

Install TabPFN-Rel for benchmarking with a local or hosted backend:

```bash
pip install "relarena[tabpfn-rel-local]"
# Or: pip install "relarena[tabpfn-rel-api]"
relarena --model tabpfn-rel-local --datasets rel-f1 --tasks driver-dnf --n-trials 1
```

This distribution depends on `relarena-core`. TabPFN-Rel is optional and can also
be installed directly as `tabpfn-rel[local]` or `tabpfn-rel[api]` for prediction on
user databases without the benchmark package.

The three distributions are sibling packages in the same monorepo. See the
[repository README](https://github.com/PriorLabs/relarena#readme) for model support,
installation extras, cache usage and benchmark commands, and the
[contributor guide](https://github.com/PriorLabs/relarena/blob/main/CONTRIBUTING.md)
for workspace development.
