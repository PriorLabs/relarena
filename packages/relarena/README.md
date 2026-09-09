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

[Documentation and contributor guide](https://github.com/PriorLabs/relarena#readme).
