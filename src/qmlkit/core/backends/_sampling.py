"""Shared sampling, so every simulator backend draws shots identically.

Qiskit's ``sample_counts`` takes no seed and Cirq's sampler carries its own RNG,
so leaving sampling to each SDK would make results irreproducible *and*
incomparable across backends. Drawing from the exact probability vector here
gives one seeded, auditable path — and it is what lets the cross-backend suite
compare sampled results at all.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sample_counts_from_probs", "normalise_probabilities"]


def normalise_probabilities(probs: np.ndarray) -> np.ndarray:
    """Clip tiny negatives from floating-point error and renormalise."""
    p = np.clip(np.asarray(probs, dtype=float).ravel(), 0.0, None)
    total = p.sum()
    if total <= 0:
        raise ValueError("probability vector sums to zero; the circuit produced no state")
    return p / total


def sample_counts_from_probs(
    probs: np.ndarray, shots: int, n_qubits: int, rng: np.random.Generator
) -> dict[str, int]:
    """Draw ``shots`` samples, keyed by ``n_qubits``-wide bitstrings.

    Qubit 0 is the most significant bit, matching the rest of the library.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    p = normalise_probabilities(probs)
    expected = 2**n_qubits
    if p.size != expected:
        raise ValueError(f"expected {expected} probabilities for {n_qubits} qubits, got {p.size}")
    draws = rng.multinomial(shots, p)
    return {format(i, f"0{n_qubits}b"): int(n) for i, n in enumerate(draws) if n > 0}
