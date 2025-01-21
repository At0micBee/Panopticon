
########################################################################################################################

# Model related modules
from .. import constants as cst     # Constant values

# Standard Python modules
import re                           # Regex
import os                           # System operations
import glob                         # Unix style name parser
import math                         # Python math
import numpy as np                  # Matrices
import pandas as pd                 # Feather format and tables
from itertools import chain         # Chaining iterators
import multiprocessing as mp        # Multiprocessing tools
import matplotlib.pyplot as pl      # Plotting module
from termcolor import colored       # Colored output

# Torch and its required subparts
import torch                        # Machine learning and tensors
from torch import Tensor            # The tensor type

FINISHED = f"[{colored('Done', color = 'green', attrs = ['bold'])}]"

########################################################################################################################
# Class for a single quarter

class Quarter:
    """
    # A quarter of observation
    """

    def __init__(self, path: str, id_sim: int, ref: pd.DataFrame) -> None:
        
        # Saving basic info
        self.path = path
        self.id_sim = id_sim
        self.file_name, _ = os.path.splitext(os.path.basename(self.path))
        self.id_quarter = int(re.search("Q([^;]*)", self.file_name).group(1))
        self.ref = ref

        # Saving the series
        self.raw = pd.read_feather(self.path).to_numpy()
        self.time = torch.tensor(self.raw[:, 0])
        self.lcs = torch.tensor(self.raw[:, 1:-1])

        # Adjusting time based on quarter, and version of PlatoSim
        if id_sim < 6700:
            self.time += (self.id_quarter - 1) * cst.Q_LENGTH_OLD
        else:
            self.time += (self.id_quarter - 1) * cst.Q_LENGTH

        # If available, we load the labels
        self.label_name = self.file_name.replace("sim", "label")
        self.base = os.path.dirname(self.path)

        head, tail = os.path.split(self.base)
        self.label_path = os.path.join(head, "labels", tail, f"{self.label_name}.pt")
        
        # If nothing we keep as None
        self.label = None
        if os.path.exists(self.label_path):
            self.label = torch.load(self.label_path)

        # Computing the passes
        self.pass_transit = self.compute_passes(
            t0 = self.ref["transit_t0"] * cst.SEC_DAY,
            period = self.ref["transit_period"] * cst.SEC_DAY
        )
        self.pass_eb = self.compute_passes(
            t0 = self.ref["EB_t0"] * cst.SEC_DAY,
            period = self.ref["EB_period"] * cst.SEC_DAY
        )
        self.pass_eb_sec = self.compute_passes(
            t0 = (self.ref["EB_t0"] - self.ref["EB_period"] / 2) * cst.SEC_DAY,
            period = self.ref["EB_period"] * cst.SEC_DAY
        )
        self.pass_beb = self.compute_passes(
            t0 = self.ref["BEB_t0"] * cst.SEC_DAY,
            period = self.ref["BEB_period"] * cst.SEC_DAY
        )
        self.pass_beb_sec = self.compute_passes(
            t0 = (self.ref["BEB_t0"] - self.ref["BEB_period"] / 2) * cst.SEC_DAY,
            period = self.ref["BEB_period"] * cst.SEC_DAY
        )
        
    def __len__(self) -> int:
        return self.lcs.shape[1]

    def __repr__(self) -> str:
        return f"Quarter {self.id_quarter}"
    
    ##################################################

    def compute_passes(self, t0: float, period: float) -> Tensor:
        """
        # Visible passes computer

        We compute for the quarter the transit events on the lc.
        """

        if math.isnan(t0):          # If there is no planet, it means we have a NaN
            return torch.tensor([])     # We return an empty tensor

        c = int((self.time[-1] - t0) / period) + 1          # Computing the max cycle
        p = torch.linspace(0, c, c + 1) * period + t0       # Computing the time of each cycle
        visible = (p > self.time[0]) & (p < self.time[-1])  # Checking if visible
        
        return p[visible]                                   # Returning the visible passes

    def make_label(self) -> Tensor:
        """
        # Creates the label for the quarter

        We use t0 and the period to see which points belong to which event,
        and return the corresponding tensor as output.

        We create the position of the classes on the time series.
        - 0: the position of the transits
        - 1: the position of the EBs
        """

        # Number of classes to find
        # 2 -> continuum, events
        # 3 -> continuum, transits, binaries
        NUM_CLASSES = 3

        # We create the time series and the corresponding classes positions
        time_scale = torch.linspace(self.time[0], self.time[-1], len(self.time), dtype = torch.float32)
        position = torch.zeros((NUM_CLASSES, len(time_scale)), dtype = torch.float32)
        
        # Computing the passes of the transits
        duration = (self.ref["transit_duration"] * cst.SEC_DAY) / 2
        for p in self.pass_transit:
            pos = (time_scale > p - duration) & (time_scale < p + duration)
            position[1, pos] = 1

        # Computing the passes of the EB
        duration = self.ref["EB_duration"] / 2
        for p in chain(self.pass_eb, self.pass_eb_sec):
            pos = (time_scale > p - duration) & (time_scale < p + duration)
            position[2, pos] = 1
        
        duration = self.ref["BEB_duration"] / 2
        for p in chain(self.pass_beb, self.pass_beb_sec):
            pos = (time_scale > p - duration) & (time_scale < p + duration)
            position[2, pos] = 1
        
        # Then we fill the continuum where no events are present
        pos_continuum = position[1, :].logical_not() | position[2, :].logical_not()
        position[0, pos_continuum] = 1

        return position

##################################################

class LightCurves:
    """
    # Plato light curve

    Combines all quarters of a simulation.
    """

    def __init__(self, id_sim: int, path_files: str, ref_df: pd.DataFrame) -> None:

        # Saving basic info
        self.id_sim = id_sim
        self.path = path_files
        self.path_figs = os.path.join(self.path, "figs")
        self.label_path = os.path.join(path_files, "labels")
        self.search_pattern = os.path.join(path_files, "*-*", f"sim{self.id_sim:05}_Q*.ftr")
        self.files = np.sort(glob.glob(self.search_pattern))

        # Creating the quarters
        self.reference = ref_df[ref_df["index_sim"] == self.id_sim].iloc[0]
        self.quarters = [ Quarter(p, id_sim, self.reference) for p in self.files ]

    def __len__(self) -> int:
        return len(self.quarters)

    def __repr__(self) -> str:
        return f"LC {self.id_sim}, {len(self.quarters)} quarters"
    
    ##################################################

    def n_cam_groups(self) -> int:
        """
        # Number of camera
        """
        return len(self.quarters[0])
    
    def draw_ax(self, ax: np.ndarray, q: Quarter) -> None:
        """
        # Unified function to draw each ax
        """
        
        if q.label is None:

            for id_cam, flux in enumerate(q.lcs.T):

                ax[id_cam].scatter(
                    q.time / cst.SEC_DAY, flux,
                    color = "black", label = f"Cam {id_cam + 1}, Q{q.id_quarter}",
                    s = 3, zorder = 0)
                ax[id_cam].legend(loc = "lower left")

                # Computing the planetary transits
                for p in q.pass_transit:
                    ax[id_cam].axvspan(
                        p / cst.SEC_DAY - self.reference["transit_duration"],
                        p / cst.SEC_DAY + self.reference["transit_duration"],
                        color = "teal"
                    )
                
                # Computing the EB
                for p in chain(q.pass_eb, q.pass_eb_sec):
                    ax[id_cam].axvspan(
                        (p - self.reference["EB_duration"]) / cst.SEC_DAY,
                        (p + self.reference["EB_duration"]) / cst.SEC_DAY,
                        color = "crimson"
                    )
        
        else:
            for id_cam, flux in enumerate(q.lcs.T):
                continuum = np.ma.masked_where(q.label[1], flux)    # We mask where events
                events = np.ma.masked_where(q.label[0], flux)       # We mask where continuum

                ax[id_cam].scatter(
                    q.time / cst.SEC_DAY, continuum,
                    color = "Teal", label = f"Cam {id_cam + 1}, Q{q.id_quarter} - Continuum",
                    s = 3, zorder = 0)

                ax[id_cam].scatter(
                    q.time / cst.SEC_DAY, events,
                    color = "Crimson", label = f"Cam {id_cam + 1}, Q{q.id_quarter} - Event",
                    s = 3, zorder = 0)
                ax[id_cam].legend(loc = "lower left")

    def plot(self) -> None:
        """
        # Plotting the full simulation

        We create a plot of all simulations and all quarters.
        """

        fig, ax = pl.subplots(
            nrows = self.n_cam_groups(), ncols = len(self),
            sharex = "col", squeeze = False, layout = "constrained",
            figsize = (2 * 6.4 * len(self), 4.8 * self.n_cam_groups())
        )

        for id_q, q in enumerate(self.quarters):
            self.draw_ax(ax = ax[:, id_q], q = q)
            
        fig.suptitle(f"Simulation {self.id_sim}")
        fig.supxlabel("Time [days]")
        fig.supylabel("Flux")

        f_name = os.path.join(self.path_figs, f"sim{self.id_sim:05}.png")
        fig.savefig(f_name, dpi = 300)
        pl.close()
    
    def make_label(self) -> None:
        """
        # For each quarter we compute the label
        """

        for q in self.quarters:

            position = q.make_label()
            save_path = os.path.join(self.label_path, f"label{self.id_sim:05}_Q{q.id_quarter}.pt")
            torch.save(obj = position, f = save_path)
            print(FINISHED, save_path)


########################################################################################################################
# Wrapper for the async function call

def _wrapper_plot(args):
    
    id_sim, path_files, ref_df = args

    sim = LightCurves(
        id_sim = id_sim,
        path_files = path_files,
        ref_df = ref_df
    )
    
    if len(sim) < 1:
        return
    
    sim.plot()

def _wrapper_label(args):

    id_sim, path_files, ref_df = args

    sim = LightCurves(
        id_sim = id_sim,
        path_files = path_files,
        ref_df = ref_df
    )

    if len(sim) < 1:
        return

    sim.make_label()

########################################################################################################################
# Function accessible from the main program

def plot(cfg: dict, plot_list: list[str]) -> None:
    """
    # Plots the given list of simulation index

    We use multiprocessing to speed computation up.
    """

    path_ref = os.path.join(cfg["path_data"], "config", "Complete_Parameters.ftr")

    path_files = cfg["path_data"]
    ref_df = pd.read_feather(path_ref)
    iterator = zip(
        iter(plot_list),
        [path_files] * len(plot_list),
        [ref_df] * len(plot_list)
    )

    num_cpu = np.min([mp.cpu_count() - 1, len(plot_list)])

    with mp.Pool(processes = num_cpu) as pool:
        pool.map(func = _wrapper_plot, iterable = iterator)

def plotall(cfg: dict) -> None:
    """
    # Creates the complete list of files to plot
    """

    path_ref = os.path.join(cfg["path_data"], "config", "Complete_Parameters.ftr")
    ref_df = pd.read_feather(path_ref)
    
    plot(cfg, ref_df["index_sim"])

def make_labels(cfg: dict) -> None:
    """
    # Creates the labels for all the files
    """

    path_ref = os.path.join(cfg["path_data"], "config", "Complete_Parameters.ftr")

    path_files = cfg["path_data"]
    ref_df = pd.read_feather(path_ref)
    iterator = zip(
        iter(ref_df["index_sim"]),
        [path_files] * len(ref_df["index_sim"]),
        [ref_df] * len(ref_df["index_sim"])
    )

    with mp.Pool(processes = cst.NUM_CPU) as pool:
        pool.map(func = _wrapper_label, iterable = iterator)

########################################################################################################################
