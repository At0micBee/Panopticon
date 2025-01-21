"""
# Utilities

This sub modules compiles some functions to manage the light curves
of the simulation.
"""

########################################################################################################################
# Calling the sub parts, simplifying calling them from the main program

from .analyzer import analyze
from .data import plot, plotall, make_labels
from .prepare import package, inspect
from .invalid import find_invalid, clean_invalid

########################################################################################################################
