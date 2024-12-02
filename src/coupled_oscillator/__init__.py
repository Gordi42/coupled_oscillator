"""
Coupled Oscillator
==================

Description
-----------
A balancing study of a simple coupled spring pendulum

For more information, visit the project's GitHub repository:
https://github.com/Gordi42/coupled_oscillator
"""
__author__ = """Silvano Gordian Rosenau"""
__email__ = 'silvano.rosenau@uni-hamburg.de'
__version__ = '0.1.0'

from typing import TYPE_CHECKING
from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules
    from . import balancing
    from . import ramp_functions

    # other imports
    from .oscillator_properties import OscillatorProperties
    from .phase import Phase
    from .hamiltonian import Hamiltonian
    from .hamiltonian_approx import HamiltonianApprox
    from .imbalance import Imbalance
    from .solver import solve, solve_for_phase

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "coupled_oscillator": ["balancing", "ramp_functions"],
}

all_imports_by_origin = {
    "coupled_oscillator.oscillator_properties": ["OscillatorProperties"],
    "coupled_oscillator.phase": ["Phase"],
    "coupled_oscillator.hamiltonian": ["Hamiltonian"],
    "coupled_oscillator.hamiltonian_approx": ["HamiltonianApprox"],
    "coupled_oscillator.imbalance": ["Imbalance"],
    "coupled_oscillator.solver": ["solve", "solve_for_phase"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)

