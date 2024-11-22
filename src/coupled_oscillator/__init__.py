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

from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules
    # from . import my_module

    # other imports
    from .oscillator_properties import OscillatorProperties
    from .phase import Phase

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    # "coupled_oscillator": ["my_module"],
}

all_imports_by_origin = {
    "coupled_oscillator.oscillator_properties": ["OscillatorProperties"],
    "coupled_oscillator.phase": ["Phase"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)

