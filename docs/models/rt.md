# RT-PluRel and RT-J

`rt-plurel` and `rt-j` are the [Relational Transformer (RT)](https://github.com/stanford-star/relational-transformer),
a relational foundation model from the [Stanford Tabular and Relational (STAR) project](https://star-project.stanford.edu),
fine-tuned per task from a published checkpoint. The two registered systems share everything but
the warm start:

| Registered identifier | Warm start | Pretraining data |
|---|---|---|
| `rt-plurel` | [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel) | [PluRel](https://github.com/stanford-star/plurel) synthetic databases |
| `rt-j` | [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) | the Join corpus of real databases |

## Source of the fine-tuning recipe

RelArena's RT integration transcribes the published fine-tuning recipe from
[`examples/finetune`](https://github.com/stanford-star/relational-transformer/tree/main/examples/finetune)
in [stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer),
as of the pinned `relational-transformer==1.10.0`: a delta fine-tune on frozen weights with
Muon, batch 256, a constant learning rate of 5e-4, weight decay 0.1, and an EMA of the weights,
run in four stages — a selection run on `train` with the step picked on `val`, a context search
on `val` by inference alone, a refit on `train + val` at the chosen step rescaled by the row
ratio, and an 8-seed test ensemble. Those are the recipes behind the fine-tuned RT-J and
RT-PluRel entries on the [RelBench leaderboard](https://github.com/stanford-star/relational-transformer/blob/main/examples/finetune/README.md).

The values RelArena runs under, and the few places its protocol has to differ from upstream
(the harness scores test, the scored split is placed past the context cutoff), are written at
the call they belong to in
[`packages/relarena/src/relarena/models/rt/config.py`](../../packages/relarena/src/relarena/models/rt/config.py)
and
[`packages/relarena/src/relarena/models/rt/model.py`](../../packages/relarena/src/relarena/models/rt/model.py).
There is no harness search space: the step and context are selected internally, so each system
produces one row with real test metrics and complete runtime.

## Installation

The pinned `relational-transformer` package provides a stable-ABI wheel for Linux x86-64, the
platform currently supported by RelArena's RT integration. A GPU is strongly recommended.

```bash
uv sync --all-packages --extra rt                 # from a source checkout
pip install "relarena[rt]"         # from a release
```

## Running RT-PluRel and RT-J

Run one task through the ordinary RelArena CLI:

```bash
uv run --all-packages --extra rt relarena \
    --model rt-j \
    --datasets rel-f1 \
    --tasks driver-dnf \
    --output rt_j_driver_dnf.csv
```

Run all 21 RelBench v1 entity classification and regression tasks by omitting `--datasets` and
`--tasks`. Substitute `--model rt-plurel` for the PluRel warm start. The model-only `--n-trials`
option does not alter the internal step and context selection.

## Citation

If you use `rt-j` or `rt-plurel`, please cite the RT-J and RT papers:

```bibtex
@inproceedings{ranjan2026rtj,
    title={{RT-J}: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions},
    author={Rishabh Ranjan and Vignesh Kothapalli and Harshvardhan Agarwal and Charilaos Kanatsoulis and Roshan Upendra and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026}
}

@inproceedings{ranjan2026relational,
    title={Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data},
    author={Rishabh Ranjan and Valter Hudovernik and Mark Znidar and Charilaos Kanatsoulis and Roshan Upendra and Mahmoud Mohammadi and Joe Meyer and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```

If you use `rt-plurel`, please also cite PluRel:

```bibtex
@inproceedings{kothapalli2026plurel,
    title={{PluRel}: Synthetic Data unlocks Scaling Laws for Relational Foundation Models},
    author={Vignesh Kothapalli and Rishabh Ranjan and Valter Hudovernik and Vijay Prakash Dwivedi and Johannes Hoffart and Carlos Guestrin and Jure Leskovec},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026}
}
```

Papers: [RT-J](https://arxiv.org/abs/2610.17867) (NeurIPS 2026),
[RT](https://arxiv.org/abs/2510.06377) (ICLR 2026),
[PluRel](https://arxiv.org/abs/2602.04029) (ICML 2026).
