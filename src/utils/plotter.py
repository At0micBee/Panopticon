"""
# Plotter script

Loads the light curves used for the training and plots them.
"""

########################################################################################################################

# Model related modules
from .. import constants as cst     # Constant values

# Standard Python modules
import os                           # System operations
import glob                         # Unix style name parser
import numpy as np                  # Matrices
import pandas as pd                 # Feather format and tables
import matplotlib.pyplot as pl      # Plotting module

########################################################################################################################
# Class for a single quarter

class PlatoLC:
    """
    # Plato light curve

    We represent the Plato light curve as a class, to be able to interact with it easily
    """

    def __init__(self, path_data: str, simulation_name: str) -> None:

        # Computing base path
        self.simulation_name = simulation_name
        self.search = os.path.join(path_data, f"{simulation_name}*.ftr")
        self.files = np.sort(glob.glob(self.search))

        self.df = []
        self.array = []
        self.time = []
        self.camera = []
        for f in self.files:
            current = pd.read_feather(f)                # Loading the file
            arr = current.values
            self.df.append( current )                   # Appending
            self.array.append( arr )                    # Getting the values of the dataframe
            self.time.append( arr[:, 0] / cst.SEC_DAY ) # Computing the time series, in days
            self.camera.append( arr[:, cst.SLICE] )     # Extracting the fluxes of the cameras

        self.col_names = self.df[0].columns
        self.num_camera = self.camera[0].shape[1]

        self.reference = pd.read_feather(f"{path_data}config/")
    
    def plot(self) -> None:
        """
        # Plots the flux from the loaded cameras

        ## Description

        Takes all the camera (or camera groups) contained in the file and plots them.
        """

        fig, ax = pl.subplots(
            nrows = self.num_camera, ncols = len(self.df),
            sharex = "col", sharey = "row",
            figsize = (6.4 * len(self.df) / 2, 4.8 * self.num_camera / 2)
        )

        if len(self.camera) == 1:
            for idx, lc in enumerate(self.camera[0].T):
                ax[idx].plot(self.time[0], lc, color = "black", label = self.col_names[idx + 1])
                ax[idx].legend(loc = "lower left")

        else:
            for q, (t, c) in enumerate(zip(self.time, self.camera)):
                for idx, lc in enumerate(c.T):
                    ax[idx, q].plot(t, lc, color = "black", label = self.col_names[idx + 1])

                    if q == 0:
                        ax[idx, q].legend(loc = "best")

        fig.suptitle(f"Light curve {self.simulation_name}")
        fig.supxlabel("Time [days]")
        fig.supylabel("Flux")

        fig.savefig(f"{cst.PATH_FIGS}/{self.simulation_name}.pdf")
        pl.close()

########################################################################################################################

def plot_lc(cfg: dict, plot_list: list[str]) -> None:
    """
    # The plotting function

    ## Description

    Takes the list of inputs (or fetches all if 'all' is the only element in the list) and plots them
    to help data visualization.

    ## Inputs

    - `cfg`: Configuration file to use
    - `plot_list`: the list of files to be plotted
    """

    # We check if we plot a list or all
    to_do = plot_list
    if len(plot_list) == 1 and plot_list[0] == "all":
        to_do = np.array(glob.glob(f"{cfg['path_data']}*.ftr"))
        to_do = np.char.rsplit(to_do, sep = "_", maxsplit = 1)
        to_do = np.unique(np.stack(to_do, axis = 0)[:,0])

    # We iterate through the list
    for n in to_do:
        f, _ = os.path.splitext(n)
        f = os.path.basename(f)
        target = PlatoLC(cfg["path_data"], f)
        target.plot()

########################################################################################################################
