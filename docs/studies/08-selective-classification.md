# Study 8 — A classifier that declines to answer

Letting a model abstain is an old and good idea: answer where you are confident,
refuse where you are not, and hand the rest to a human. In quantum machine learning
it arrives with an extra motive — a quantum model's output is a *distribution*, so
reading it costs repeated circuit executions, and a rule that accepts a prediction
early saves shots as well as errors.

It also produces the most flattering number in classification, and the flattery is
free. This study builds an abstaining classifier, reports the number that looks
good, and then shows why it is the wrong one to quote.

## The trap, stated first

Selective accuracy — accuracy on the samples the model chose to answer — rises
monotonically as you abstain more. Push the threshold far enough and it reaches
1.000 on the single sample you are surest about. So a selective accuracy quoted
against a model that answered *everything* is not a comparison between two models;
it is a comparison between two different questions.

```python
# docs: requires sklearn
import numpy as np
import qmlkit as qk

truth = np.array([0, 1, 0, 1, 0, 1, 0, 1])
answered_only_when_certain = np.array([0, 1, None, None, None, None, None, None],
                                      dtype=object)

scores = qk.evaluate.selective(truth, answered_only_when_certain)
print(f"selective_accuracy {scores['selective_accuracy']:.3f} "
      f"at coverage {scores['coverage']:.0%}")
print(f"accuracy           {scores['accuracy']:.3f}   <- comparable to a model that cannot abstain")
for note in scores.notes:
    print("note:", note)
```

A perfect selective score, bought with coverage. `qk.evaluate.selective` reports
both halves and makes the comparable one the **primary** metric, so
`scores.score` cannot quietly become the flattering number.

## On a real model

Breast cancer, three PCA components, a two-layer `VQC`. The classical bar first, as
always:

```python
# docs: requires sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X = PCA(n_components=3, random_state=0).fit_transform(StandardScaler().fit_transform(X))
X = qk.AngleScaler().fit_transform(X)
print(f"{len(X)} samples, {X.shape[1]} features")
```

Training a `VQC` on 455 rows and evaluating on 114 held out, the model's own
confidence — the larger of its two class probabilities — becomes the abstention
signal. Sweeping the threshold gives this, measured:

| threshold | selective accuracy | coverage | **comparable accuracy** |
|---|---|---|---|
| answer everything | 0.8947 | 100% | **0.8947** |
| 0.60 | 0.9182 | 96.5% | 0.8860 |
| 0.70 | 0.9320 | 90.4% | 0.8421 |
| 0.80 | **0.9479** | 84.2% | **0.7982** |

Read the two shaded columns together. As the threshold rises, selective accuracy
climbs from 0.895 to **0.948** — a 5.3-point gain that would headline well. Over the
same sweep the comparable accuracy *falls* from 0.895 to **0.798**.

Abstention did not make this model better. It made the reported number better and
the model worse, and only one of the two columns says so.

For reference, `qk.baseline` puts `svc-linear` at **0.9507** on the same three
features — above every row in the table, including the flattering one.

## The curve, because the point was chosen

A single `(coverage, accuracy)` pair is a threshold someone picked. The whole trade
is a curve, and it is what makes two abstaining models comparable:

```python
# docs: requires sklearn
rng = np.random.default_rng(0)
labels = rng.integers(0, 2, 200)
confidence = rng.uniform(0, 1, 200)
predictions = np.where(confidence > 0.35, labels, 1 - labels)   # right when confident

curve = qk.evaluate.risk_coverage(labels, predictions, confidence, n_points=5)
for c, r in zip(curve["coverage"], curve["risk"]):
    print(f"  coverage {c:.2f}   risk {r:.3f}")
print(f"AURC {curve['aurc']:.4f}")
```

**AURC** — area under the risk–coverage curve — is the summary that abstaining more
cannot inflate. A model that buys a high selective accuracy by refusing most of the
data pays for it in the low-coverage region, where its risk is measured against the
handful of samples it kept. On the `VQC` above the curve is `(0.25, 0.000)`,
`(0.50, 0.000)`, `(0.75, 0.024)`, `(1.00, 0.105)` with **AURC 0.0198**: the
confidence really is informative — errors concentrate where the model is unsure —
which is what justifies abstaining at all, and is a separate question from whether
abstaining helped.

## Which number to quote

- Comparing against a model that **cannot** abstain — quote `accuracy`. It counts an
  abstention as an error, which is what an abstention costs you when something
  downstream needs an answer.
- Comparing two abstaining models — quote **AURC**, or fix a coverage and compare
  selective accuracy *there*. Comparing at two different coverages compares nothing.
- Reporting the saving in circuit executions — say what it bought and what it cost,
  in the same sentence. `qk.plan()` prices the run; the execution saving at
  prediction time is usually small beside the training budget that produced the model.

## Where this comes from

The construction is classical — Chow's reject option, 1970 — and recurs in quantum
machine learning because a quantum prediction is already a repeated measurement. For
a recent treatment in this setting, and the measurement rule this study's shape is
drawn from, see Ptáček, Lewandowska and Kukulski, *Resource-Efficient Variational
Quantum Classifier*, [arXiv:2511.09204](https://arxiv.org/abs/2511.09204). Their
construction discards shots whose qubits disagree and votes on the survivors, which
is abstention implemented in the readout rather than in post-processing.

Quoting any selective result, theirs or your own, needs the coverage beside it. That
is the whole argument for `qk.evaluate.selective` reporting both.

---

**Next:** back to the [study index](index.md), or the
[evaluation guide](../guides/evaluation.md) for the metrics this one builds on.
