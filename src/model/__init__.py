"""
# Panopticon

The building blocks for the model and its dependencies.
"""

########################################################################################################################
# Calling the sub parts, simplifying calling them from the main program

from .loss import Loss                  # The loss function
from .functions import Run              # Simulation running tool
from .dataset import LightCurveSet      # The dataset for the PLATO light curves

from .selector import optimizer         # Selector of optimizer and schedule

########################################################################################################################
