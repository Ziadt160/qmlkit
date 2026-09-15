# Guides

The tutorials show you how. These explain why, and are meant to be read out of order
when a particular question comes up.

| | |
|---|---|
| **[Coming from PennyLane](from-pennylane.md)** | Borrow the three loops worth borrowing without migrating anything, the name map, and the four places the two libraries genuinely differ |
| **[The parameter-shift rule](parameter-shift.md)** | Why the famous two-term formula is not the whole story, and how qmlkit derives a rule per gate instead of transcribing one |
| **[Choosing a gradient method](choosing-a-gradient.md)** | Six options, a decision table, and measured costs |
| **[Backends and conventions](backends.md)** | Endianness, controlled-gate ordering, precision floors, and three upstream discrepancies worth knowing about |
| **[Integrating OpenQARP](openqarp.md)** | What qmlkit takes from Fujitsu's C++ engine, how the translation works, and the measurements that decide when to reach for it |
| **[Running under noise](noise.md)** | Mixed-state backends, why noise never picks a simulator for you, and what a noise model does to a gradient |
| **[Extending qmlkit](extending.md)** | Adding a gate, an ansatz, a gradient estimator or a backend — every extension point is a registry |
| **[Evaluating a model honestly](evaluation.md)** | Metrics that say when they mislead, skewed classes, the classical bar, and whether a number can be trusted or reproduced |
| **[Watching a run](watching-a-run.md)** | A live line with an honest ETA, what the run spent its time on, and the whole thing written out as one self-contained page |
| **[Working with a coding agent](agents.md)** | Why a wrong name answers with the right one, and what `diagnose()` catches that nothing raises for |

For how any of this is known to be correct, see [Validation](../about/validation.md).
