r"""Data re-uploading, as a pattern rather than a fixed structure.

Re-uploading is *any* interleaving of an encoding with a trainable block. The
encoding can be any feature map, the trainable block any ansatz block, and the
order, depth and sharing are all free. Treating it as one hardcoded class is a
category error — so it is not one here.

:func:`reupload` is a convenience over that freedom, not a replacement for it:

    reupload(fmap, n_layers=3)                                   # S W S W S W
    reupload(fmap, n_layers=3, order="WS")                       # W S W S W S
    reupload(fmap, n_layers=3, block=RotationLayer("ry") + EntanglerLayer("cz", "ring"))
    reupload(fmap, n_layers=3, share_weights=True)               # one tied block, reused

Anything it cannot express, compose directly — that is the same vocabulary:

    Ansatz(n, EncodingLayer(zz) + RotationLayer("ry") + EncodingLayer(angle))

**Frequencies.** ``L`` uploads reach frequencies ``0..L`` only when the trainable
block does not commute with the encoding rotation; if it does, the uploads merge
into a single rotation. :func:`reupload` checks this and warns.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from qmlkit.ansatz.blocks import (
    Block,
    EncodingLayer,
    EntanglerLayer,
    RotationLayer,
    repeat,
    share,
)
from qmlkit.ansatz.library import Ansatz
from qmlkit.utils.errors import unknown

__all__ = ["reupload", "ReuploadModel"]


def _commutes_with_encoding(block: Block, feature_map: object) -> set[str] | None:
    """The trainable gates when the uploads really do collapse, else ``None``.

    ``Ry(x) Ry(t1) Ry(x) Ry(t2) = Ry(2x + t1 + t2)`` — the uploads collapse into one
    rotation, the model reaches a single frequency, and the weights become a phase.

    **That identity holds on one wire with nothing in between.** Entanglement breaks
    it, so an ``EntanglerLayer`` anywhere in the block, or a feature map that entangles
    on its own, means there is nothing to warn about — the model reaches ``0..2L``
    rather than one frequency, and saying otherwise sends the caller to redesign a
    circuit that was correct.

    Returns the gates rather than a bool so the warning can name what it actually
    found, instead of the ``rotations`` parameter the caller may never have used.
    """
    encoding = getattr(feature_map, "rotation", None)
    if encoding is None:
        return None  # a multi-gate map (ZZ, Pauli) never fully commutes
    if getattr(feature_map, "entangle", False):
        return None  # the encoding entangles, so the uploads cannot merge per wire
    gates: set[str] = set()
    entangled = False

    def walk(b: Block) -> None:
        nonlocal entangled
        if isinstance(b, EntanglerLayer):
            entangled = True
        if isinstance(b, RotationLayer):
            gates.update(b.gates)
        for attr in ("blocks", "block"):
            child = getattr(b, attr, None)
            if isinstance(child, Block):
                walk(child)
            elif isinstance(child, tuple):
                for c in child:
                    walk(c)

    walk(block)
    if entangled or not gates or not gates <= {encoding}:
        return None
    return gates


def reupload(
    feature_map: object,
    n_layers: int = 3,
    block: Block | None = None,
    order: str = "SW",
    entangler: str | None = "cx",
    pattern: str = "chain",
    rotations: Sequence[str] = ("rz", "ry", "rz"),
    share_weights: bool = False,
    name: str = "reupload",
) -> Ansatz:
    """Build a re-uploading ansatz from any feature map and any trainable block.

    Parameters
    ----------
    feature_map
        Any :class:`~qmlkit.encoding.feature_maps.FeatureMap`.
    block
        The trainable block. Defaults to a rotation layer plus an entangler.
    order
        ``"SW"`` encodes then varies; ``"WS"`` varies then encodes. The difference
        is real: ``"WS"`` lets the model transform the state before the first
        upload, ``"SW"`` does not.
    share_weights
        Tie one trainable block across every layer — far fewer parameters, and the
        gradient sums over occurrences.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if order not in ("SW", "WS"):
        raise unknown(
            "order",
            order,
            ("SW", "WS"),
            hint="'SW' encodes then varies; 'WS' varies before the first upload.",
        )

    n_qubits = int(feature_map.n_qubits)  # type: ignore[attr-defined]
    n_inputs = int(feature_map.n_angles)  # type: ignore[attr-defined]

    if block is None:
        block = RotationLayer(rotations)
        if entangler and n_qubits > 1:
            block = block + EntanglerLayer(entangler, pattern)

    collapsing = _commutes_with_encoding(block, feature_map)
    if collapsing is not None:
        found = ", ".join(repr(g) for g in sorted(collapsing))
        encoding_gate = getattr(feature_map, "rotation", "the encoding")
        warnings.warn(
            f"the trainable block only uses {found}, which commutes with the "
            f"{encoding_gate!r} encoding rotation: the uploads collapse into a single "
            f"rotation, so the model reaches one frequency instead of 0..{n_layers} and "
            "its weights do nothing beyond a phase. Use a block whose generators differ "
            "from the encoding's, or add an entangler, either of which breaks the "
            "collapse.",
            UserWarning,
            stacklevel=2,
        )

    encoding = EncodingLayer(feature_map)
    layer = (encoding + block) if order == "SW" else (block + encoding)
    body = share(n_layers, layer) if share_weights else repeat(n_layers, layer)
    return ReuploadModel(n_qubits, body, name, n_inputs, feature_map, n_layers)


class ReuploadModel(Ansatz):
    """An :class:`Ansatz` that also knows its encoding and upload count."""

    def __init__(
        self,
        n_qubits: int,
        block: Block,
        name: str,
        n_inputs: int,
        feature_map: object,
        n_uploads: int,
    ) -> None:
        super().__init__(n_qubits, block, name, n_inputs)
        self.feature_map = feature_map
        self.n_uploads = n_uploads

    @property
    def n_frequencies(self) -> int:
        """Reachable frequencies ``0..L``, so ``L + 1`` of them."""
        return self.n_uploads + 1

    def __repr__(self) -> str:
        return (
            f"ReuploadModel({self.name!r}, n_qubits={self.n_qubits}, "
            f"uploads={self.n_uploads}, inputs={self.n_inputs}, weights={self.n_weights})"
        )
