"""A dense reference simulator that shares nothing with qmlkit.

Written from scratch against the documented conventions - qubit 0 is the most
significant bit, ops apply left to right - and it reads **only** ``spec.ops``. Every
gate matrix is written out by hand here and every derivative matrix is hand-derived;
it calls no qmlkit execution code, no backend, and no gradient routine.

That independence is the whole point. The cross-backend suite proves the five backends
agree with each other, and the PennyLane parity suite proves qmlkit agrees with another
library - but two implementations that inherited the same convention are wrong together
without either noticing. This one inherited nothing.

It came out of an adversarial audit that used it to find a real defect: the torch
backend was reimplementing ``np.moveaxis`` without numpy's ``sorted(zip(...))``, so a
two-qubit gate on descending wires permuted the *untouched* wires. Four other
references agreed with each other and with the bug's victim only because none of them
was independent of it.

Kept deliberately small and slow. It is a referee, not a simulator - if it ever needs
optimising, it has stopped being one.
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)
Hm = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _rot(P, t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * P


def _drot(P, t):
    return -0.5 * np.sin(t / 2) * I2 - 0.5j * np.cos(t / 2) * P


def _ctrl(u):
    out = np.eye(4, dtype=complex)
    out[2:, 2:] = u
    return out


def _dctrl(du):
    out = np.zeros((4, 4), dtype=complex)
    out[2:, 2:] = du
    return out


MATS = {
    "i": lambda: I2,
    "x": lambda: Xm,
    "y": lambda: Ym,
    "z": lambda: Zm,
    "h": lambda: Hm,
    "s": lambda: np.diag([1, 1j]).astype(complex),
    "sdg": lambda: np.diag([1, -1j]).astype(complex),
    "t": lambda: np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex),
    "tdg": lambda: np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex),
    "cx": lambda: _ctrl(Xm),
    "cy": lambda: _ctrl(Ym),
    "cz": lambda: _ctrl(Zm),
    "swap": lambda: np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    ),
    "rx": lambda t: _rot(Xm, t),
    "ry": lambda t: _rot(Ym, t),
    "rz": lambda t: _rot(Zm, t),
    "phase": lambda t: np.diag([1, np.exp(1j * t)]).astype(complex),
    "crx": lambda t: _ctrl(_rot(Xm, t)),
    "cry": lambda t: _ctrl(_rot(Ym, t)),
    "crz": lambda t: _ctrl(_rot(Zm, t)),
}

DMATS = {
    "rx": lambda t: _drot(Xm, t),
    "ry": lambda t: _drot(Ym, t),
    "rz": lambda t: _drot(Zm, t),
    "phase": lambda t: np.diag([0, 1j * np.exp(1j * t)]).astype(complex),
    "crx": lambda t: _dctrl(_drot(Xm, t)),
    "cry": lambda t: _dctrl(_drot(Ym, t)),
    "crz": lambda t: _dctrl(_drot(Zm, t)),
}


def embed(u, qubits, n):
    """Embed a k-qubit matrix acting on `qubits` (big-endian labels) into 2^n."""
    k = len(qubits)
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    for row in range(dim):
        rbits = [(row >> (n - 1 - q)) & 1 for q in range(n)]
        ridx = 0
        for i, q in enumerate(qubits):
            ridx |= rbits[q] << (k - 1 - i)
        for cs in range(2**k):
            cbits = list(rbits)
            for i, q in enumerate(qubits):
                cbits[q] = (cs >> (k - 1 - i)) & 1
            col = 0
            for q in range(n):
                col |= cbits[q] << (n - 1 - q)
            out[row, col] = u[ridx, cs]
    return out


def angle(p, theta):
    """Resolve a param entry (float or ParamRef) to a number."""
    if hasattr(p, "index"):
        return float(p.scale) * float(theta[p.index]) + float(p.offset)
    return float(p)


def op_matrices(spec, theta):
    """List of full-space unitaries in application order."""
    n = spec.n_qubits
    mats = []
    for op in spec.ops:
        f = MATS[op.gate]
        u = f(*[angle(p, theta) for p in op.params]) if op.params else f()
        mats.append(embed(u, list(op.qubits), n))
    return mats


def unitary(spec, theta=None):
    n = spec.n_qubits
    U = np.eye(2**n, dtype=complex)
    for m in op_matrices(spec, theta):
        U = m @ U
    return U


def state(spec, theta=None):
    n = spec.n_qubits
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    for m in op_matrices(spec, theta):
        psi = m @ psi
    return psi


PAULI = {"I": I2, "X": Xm, "Y": Ym, "Z": Zm}


def obs_matrix(obs, n):
    """Dense matrix for a qmlkit PauliSum / PauliString."""
    terms = getattr(obs, "terms", None)
    if terms is None:
        terms = [obs]
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    for t in terms:
        coeff = complex(t.coeff)
        m = np.eye(dim, dtype=complex)
        for q, p in t.paulis:
            m = m @ embed(PAULI[p.upper()], [q], n)
        out = out + coeff * m
    return out


def expval(spec, obs, theta=None):
    psi = state(spec, theta)
    O = obs_matrix(obs, spec.n_qubits)
    return float(np.real(np.conj(psi) @ (O @ psi)))


def grad(spec, theta, obs):
    """Exact analytic gradient via the product rule over slots.

    dE/dtheta_k = sum over slots referencing k of scale * 2 Re<dpsi|O|psi>.
    """
    theta = np.asarray(theta, dtype=float)
    n = spec.n_qubits
    mats = op_matrices(spec, theta)
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0
    O = obs_matrix(obs, n)

    prefix = [psi0]
    for m in mats:
        prefix.append(m @ prefix[-1])
    psi = prefix[-1]

    out = np.zeros(len(theta))
    for oi, op in enumerate(spec.ops):
        if not op.params:
            continue
        for p in op.params:
            if not hasattr(p, "index"):
                continue
            k = p.index
            vals = [angle(pp, theta) for pp in op.params]
            du = DMATS[op.gate](*vals)
            dU = embed(du, list(op.qubits), n)
            v = dU @ prefix[oi]
            for m in mats[oi + 1 :]:
                v = m @ v
            out[k] += float(p.scale) * 2.0 * np.real(np.conj(v) @ (O @ psi))
    return out


def probabilities(spec, theta=None):
    psi = state(spec, theta)
    return np.abs(psi) ** 2
