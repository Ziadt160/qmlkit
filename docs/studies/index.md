# Case studies

The tutorials show how each piece works. These show a whole problem worked through,
from the raw data to a number you could defend — and in most of them the number is
that the quantum model **lost**.

That is deliberate. A library that only publishes its wins teaches you nothing about
when to trust it, and every case study here ends with the verdict the library itself
prints rather than the one the author would have preferred.

| Study | Task | What it is really about |
|---|---|---|
| [1. Imbalanced classification](01-imbalanced-classification.md) | 32,581 loan applications, 21.8% default | The metric that lies, and the one keyword that fixes the model |
| [2. Is a quantum kernel worth it?](02-quantum-kernels.md) | Kernel methods on small data | Two diagnostics that point opposite ways, answered before fitting |
| [3. Regression](03-regression.md) | A smooth non-linear target | Why `r2` and `rmse` disagree, and what re-uploading buys |
| [4. Chemistry: H₂ ground state](04-chemistry.md) | VQE to chemical accuracy | An ansatz that converges confidently to the wrong energy |
| [5. Clustering and generative models](05-beyond-classification.md) | Unsupervised, and a Born machine | The metrics that exist because accuracy does not apply |

Every code block on these pages runs in CI (`tests/test_docs.py`), so the numbers are
produced by the code beside them and cannot drift from it.

## The shape they all share

Each study follows the same five questions, because they are the questions that decide
whether a result means anything:

1. **What will the data break?** — `imbalance_report`, and the split it implies
2. **What is the bar?** — `baseline`, run *before* any quantum code
3. **Is the model quietly broken?** — `diagnose`, before training rather than after
4. **What does the metric actually say?** — `evaluate`, with its notes
5. **Is the number right, and reproducible?** — `selfcheck` and `fingerprint`

The heavier versions live in `examples/` — [`credit_risk.py`](https://github.com/Ziadt160/qmlkit/blob/main/examples/credit_risk.py)
is the full twelve-step version of study 1 on the real Kaggle table, and
[`experiments.py`](https://github.com/Ziadt160/qmlkit/blob/main/examples/experiments.py)
runs H₂, MNIST and breast cancer at full size.
