# Study 5 — Clustering and generative models

Accuracy needs labels and a decision. Two large parts of machine learning have
neither, and they fail in ways a classification metric cannot describe: a clustering
that is beautifully separated and answers the wrong question, and a generative model
that assigns zero probability to something that actually happens.

## Clustering: two kinds of "good" that routinely disagree

```python
import numpy as np
import qmlkit as qk

X, y = qk.datasets.make_blobs(n_samples=90, centers=3, seed=0)
labels = qk.algorithms.QMeans(n_clusters=3, seed=0).fit_predict(X)

scores = qk.evaluate.clustering(X, labels, y_true=y)
print(scores)
```

`qk.evaluate.clustering` returns both families, because they answer different
questions and you usually only have one of them:

**Internal** — no ground truth needed. `silhouette` asks whether points sit closer to
their own cluster than to the next one; `davies_bouldin` compares within-cluster
spread to between-cluster distance (lower is better). These say the partition is
*clean*.

**External** — needs labels. `adjusted_rand` corrects for chance agreement, so zero
means "no better than a random partition of the same shape"; `normalized_mutual_info`
measures shared information; `purity` is the honest-but-flattering one, since it rises
automatically as clusters get smaller.

They disagree constantly. A silhouette of 0.7 with an ARI of 0.05 is a well-separated
partition of something other than what you were looking for — and that is a finding,
not a failure, but only if you looked at both.

```python
print(f"cluster sizes {scores.extras['cluster_sizes']}")
```

The primary metric is `adjusted_rand` when labels are given and `silhouette` when they
are not, so the number you quote does not silently change meaning with the arguments.

## Generative: the model that assigns zero to something real

A quantum circuit Born machine samples bitstrings from `|psi|^2`. The standard target
is bars-and-stripes — six valid patterns out of sixteen four-bit strings:

```python
patterns = qk.datasets.bars_and_stripes(2)
target = np.zeros(16)
for row in patterns:
    target[int("".join(map(str, row.astype(int))), 2)] += 1
target /= target.sum()

uniform = np.full(16, 1 / 16)
print(f"a uniform guess scores TV = "
      f"{qk.evaluate.generative(uniform, target)['total_variation']:.3f}")
```

That is the number to beat. Reporting a total variation of 0.5 means nothing until you
know that guessing scores 0.625.

```python
model = qk.generative.QCBM(n_qubits=4, n_layers=3, seed=0)
model.fit(patterns, n_iterations=60)

scores = qk.evaluate.generative(model.probabilities(), target)
print(scores)
```

### Why total variation is the primary, and KL is not

```python
missing = qk.evaluate.generative(np.eye(16)[0], target)
print(f"KL(target || model) = {missing['kl_target_model']}")
print(f"total variation     = {missing['total_variation']:.3f}")
```

A model that puts zero mass where the target has some gives an **infinite** KL. That
is mathematically correct and useless as a training signal — it is infinite for a model
that misses one rare pattern and equally infinite for one that has learned nothing.

Total variation stays finite, is bounded in `[0, 1]`, and is a metric. `hellinger` and
`js_distance` share those properties. `support_coverage` reports the fraction of the
target's support the model reaches at all, and a note fires whenever mass is missing,
so the infinity is explained rather than just printed.

All divergences here are in **nats**, stated because half the literature uses bits and
the factor of `ln 2` is exactly the kind of silent discrepancy this library exists to
avoid.

## The verdict

A four-qubit QCBM at this depth and iteration budget beats a uniform guess and does not
master the distribution. Depth helps — eight layers reaches roughly `TV = 0.49` against
the uniform `0.625`, with about half its mass on valid patterns against the `0.375` a
random distribution puts there — but that is a modest win on a sixteen-outcome problem
a lookup table solves exactly.

Which is the point of measuring it against the uniform baseline rather than against
zero. `total_variation = 0.49` sounds like a result; `0.49 against a 0.625 floor`
is one.
