"""Core geometry primitives for Riemannian Energy Matching."""

from .geometry import (
    BoundedScalarMobility,
    ConstantDiagonalMobility,
    DiagonalPlusLowRankMobility,
    DiagonalMobility,
    GaugeFixedDiagonalMobility,
    GaugeFixedFullMobility,
    IdentityMobility,
    UnfixedLogDiagonalMobility,
    construct_spd_mapping,
    descent_condition_diagnostics,
    hutchinson_divergence,
    minimal_distortion_regularizer,
    mobility_diagnostics,
    mobility_scale_regularizer,
    riemannian_langevin_step,
    riemannian_transport_loss,
    riemannian_velocity,
)
from .networks import REMModel, build_mobility

__all__ = [
    "BoundedScalarMobility",
    "ConstantDiagonalMobility",
    "DiagonalPlusLowRankMobility",
    "DiagonalMobility",
    "GaugeFixedDiagonalMobility",
    "GaugeFixedFullMobility",
    "IdentityMobility",
    "UnfixedLogDiagonalMobility",
    "construct_spd_mapping",
    "descent_condition_diagnostics",
    "hutchinson_divergence",
    "minimal_distortion_regularizer",
    "mobility_diagnostics",
    "mobility_scale_regularizer",
    "riemannian_langevin_step",
    "riemannian_transport_loss",
    "riemannian_velocity",
    "REMModel",
    "build_mobility",
]
