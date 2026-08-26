# Tutorials

Eight pages, in order. Each one builds on the last, and each is short enough to read
in a sitting.

**Every Python snippet on these pages is executed by the test suite.** They are not
illustrations of the API — they are tests of it, so a rename that breaks a tutorial
breaks the build. The outputs shown were produced by running the code, not written by
hand.

<div class="grid cards" markdown>

- **[1. Circuits are data](01-first-circuit.md)**

    Build, draw, run, measure. The `CircuitSpec`/slot idea that everything else
    depends on, and why shots come with an error bar.

- **[2. Getting data in](02-encoding-data.md)**

    Angle, basis, amplitude and Pauli feature maps — the choice that fixes what your
    model can represent, before training starts.

- **[3. Gradients](03-gradients.md)**

    Six methods, one function. Plus the two ways a hand-rolled parameter-shift
    returns a smooth, plausible, wrong answer.

- **[4. Designing an ansatz](04-ansatz-design.md)**

    A block vocabulary where a new ansatz is one line and inherits gradients,
    resource counting and a torch layer for free.

- **[5. Training with PyTorch](05-training-torch.md)**

    `VQC` in two lines, `QuantumLayer` in any `nn.Module`, and the input gradient
    that decides whether your classical pre-net trains at all.

- **[6. Quantum kernels](06-quantum-kernels.md)**

    No variational training, a convex solver — and a quadratic circuit cost plus
    exponential concentration waiting for you.

- **[7. Re-uploading and Fourier](07-reuploading.md)**

    Why depth buys frequencies, measured rather than asserted, and the commuting
    block that silently collapses the whole model.

- **[8. Trainability](08-trainability.md)**

    Barren plateaus, cost locality, and three optimisers that exploit structure a
    general-purpose one cannot see.

</div>

## Running them yourself

Everything except tutorials 5 and 6 needs only a bare install:

```bash
pip install qmlkit
```

Tutorial 5 needs `qmlkit[torch]`, and the classification section of tutorial 6 needs
`qmlkit[sklearn]`. Both pages say so where it matters.

If you would rather have one script than eight pages, the repository ships
`examples/quickstart.py`, which walks the same ground end to end.
