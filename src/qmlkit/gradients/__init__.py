"""Gradients: shift rules, adjoint differentiation, SPSA, and one dispatcher."""

from qmlkit.gradients.adjoint import adjoint_grad, supports_adjoint
from qmlkit.gradients.dispatch import (
    choose_method,
    grad,
    gradient_cost,
    hessian,
    list_gradient_methods,
    register_gradient,
)
from qmlkit.gradients.hadamard import (
    hadamard_grad,
    hadamard_grad_cost,
    supports_hadamard_grad,
)
from qmlkit.gradients.parameter_shift import (
    finite_diff_grad,
    grad_circuit_cost,
    param_shift_grad,
    param_shift_grad_circuit,
)
from qmlkit.gradients.rules import (
    ShiftRule,
    four_term_rule,
    general_shift_rule,
    rule_for_frequencies,
    rule_for_gate,
    second_derivative_rule,
    two_term_rule,
)
from qmlkit.gradients.spsa import SPSASchedule, minimize_spsa, spsa_grad, spsa_step

__all__ = [
    "grad",
    "choose_method",
    "register_gradient",
    "list_gradient_methods",
    "adjoint_grad",
    "hadamard_grad",
    "hadamard_grad_cost",
    "supports_hadamard_grad",
    "hessian",
    "gradient_cost",
    "supports_adjoint",
    "param_shift_grad",
    "param_shift_grad_circuit",
    "grad_circuit_cost",
    "finite_diff_grad",
    "spsa_grad",
    "spsa_step",
    "SPSASchedule",
    "minimize_spsa",
    "ShiftRule",
    "general_shift_rule",
    "rule_for_frequencies",
    "rule_for_gate",
    "two_term_rule",
    "four_term_rule",
    "second_derivative_rule",
]
