"""
# Constants
"""

########################################################################################################################

# Standard Python modules
import torch
from os import environ                          # Environment operations
from os.path import dirname, join, realpath     # Path operations
from multiprocessing import cpu_count           # Multi-process cpu count

########################################################################################################################
# Getting the location of the required files

_CLUSTER_CPU = environ.get("SLURM_JOB_CPUS_PER_NODE")
NUM_CPU = _CLUSTER_CPU if _CLUSTER_CPU is not None else cpu_count()
"""
# Get the number of available cpu

Checks if we are on the cluster, falls back to default methods if necessary.
"""

PATH_LOCAL = dirname(__file__)
"""
# Path to currently running file

It is computed using `os.path.dirname(__file__)`, for compatibility.
"""

PATH_CONFIG = realpath(join(PATH_LOCAL, "..", "config", "default.toml"))
"""
# Configuration file path

Computed from the `PATH_LOCAL` value. The path to the default config file.
If none are specified when running the program, this is the one used.
"""

PATH_OUT = realpath(join(PATH_LOCAL, "..", "out"))
"""
# Output path

Computed from the `PATH_LOCAL` value. The root used to save the results of the model.
A sub folder will be generated for each run, based on the start time of the program.
"""

PATH_FIGS = realpath(join(PATH_LOCAL, "..", "figs"))
"""
# Generated figures path

Computed from the `PATH_LOCAL` value. Where the generated figures are saved.
"""

PATH_PREPARED = realpath("/data/Plato_sim_data/")
"""
# The path to where all the data is stored
"""

########################################################################################################################
# Data analysis related

SLICE = slice(1, -1)
"""
# The slice used to obtain the flux of the cameras in the Dataframes.

So far, we need to skip the zeroth column (`time`), and the last (`coaddition`). If the slice becomes
more verbose in a future version, it seems easier to have a single instance to modify.
"""

SIMS_PARAMS_PREP = [
    # Host
    "radius_star",
    "mag", "Teff",
    "logg", "logRHK",
    "Prot", "Pmin", "Pmax",
    "lmax", "Pcyc", "Povl", "Acyc",
    "numax", "deltanu",

    # Planet
    "radius_planet",
    "transit_period",
    "transit_duration",
    "transit_depth",
    "a", "e", "ip", "w",

    # Binary
    "EB_t0", "EB_period", "EB_duration",

    # Background binary
    "BEB_t0", "BEB_period", "BEB_duration"
]
"""
# Basic simulation parameters to put in the reference file

The parameters to add to the reference file, other than the light curve and the reference classes.
"""

SIMS_PARAMS = SIMS_PARAMS_PREP + ["transit_snr"]
"""
# Additional simulation parameters to put in the reference file

"""

THRESHOLDS = torch.arange(0.05, 0.99, 0.05)
"""
# The detection thresholds

The cutoff for the validation detection.
"""

THRESHOLD_BOX = 0.5
"""
# Box IOU threshold

We check if the detected box IOU is greater than this value.
"""

STD_DURATION = 1200
"""
# Duration of a TCE

The delta is in number of points in the LC, as of the writing of the code, the LC have
a 1 minute sampling, so it translates to minutes. If the sampling changes, please be aware
that this value needs to change!

1200 = 20 hours
"""

SEC_DAY = 3600 * 24
"""
# Seconds in a day

Used to scale plots to be more readable.
"""

Q_POINT_TIME = 1.5
"""
# Time to point between quarters

Value is in days.
"""

Q_LENGTH = SEC_DAY * (91.310549769 - Q_POINT_TIME)
"""
# Quarter length

New version of PlatoSim.
"""

Q_LENGTH_OLD = SEC_DAY * (90 - Q_POINT_TIME)
"""
# Quarter length

Old version of PlatoSim.
"""

POINTS_KEEP = slice(0, 126_720)
"""
# Points to keep the same length of data.

Because of the switch of the PlatoSim version, we have to be careful to not load all the points
produced by the new version, which changed the duration of quarters. We instead keep the same
number of points (and therefore the same duration) as the old file, that is 126720 points (90 days).
"""

########################################################################################################################
