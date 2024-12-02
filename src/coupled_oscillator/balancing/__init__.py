"""
Balancing
=========

Description
-----------
This module holds the various classes related to project a state of the
coupled oscillator system to a balanced state. A balanced state is a state
where the fast oscillations of the spring are eliminated.
"""
from typing import TYPE_CHECKING
from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules
    # from . import my_module

    # other imports
    from .base import BalanceBase

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    # "coupled_oscillator": ["my_module"],
}

all_imports_by_origin = {
    "coupled_oscillator.balancing.base": ["BalanceBase"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)

