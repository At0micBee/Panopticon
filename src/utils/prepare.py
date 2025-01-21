"""
# Augmenting the data to be used for training

Note that the creation of labels has to have happened previously.
"""

########################################################################################################################

# Model related modules
from .. import constants as cst         # Constant values

import os                               # System operations
import glob                             # TOML file parser
import numpy as np                      # Mathematics operations
import pandas as pd                     # Dataframes
import multiprocessing as mp        # Multiprocessing tools
import matplotlib.pyplot as pl          # Plotting tools
import scipy.interpolate as interp      # Interpolation function
from termcolor import colored       # Colored output

# Torch and its required subparts
import torch                            # Machine learning and tensors

########################################################################################################################

PACKAGING = f"[{colored('Packaging', color = 'green', attrs = ['bold'])}]"
FAILED = f"[{colored('Failed', color = 'red', attrs = ['bold'])}]"
ORIGINAL = colored('Original', color = 'blue', attrs = ['bold'])
AUGMENTED = colored('Augmented', color = 'yellow', attrs = ['bold'])

ORDER_K = 2         # Only a 2 deg is enough
SMOOTHING = 1e12    # Very large smoothing sample

########################################################################################################################

def inspect_lc(root: str, lc_name: int) -> None:
    """
    # Plots the prepared version of the LCs

    Fetches the original lc file, and checks whether there is an augmented version. Plots the
    the results to be visually inspected.
    """

    def _plot_ax(a, time, data) -> None:
        
        mask_transit = data["ref"][1].to(dtype = bool)
        continuum = np.ma.masked_array(data["lc"], mask_transit)
        transits = np.ma.masked_array(data["lc"], ~mask_transit)

        a.scatter(time, continuum, c = "teal", s = 2, label = "Continuum")
        a.scatter(time, transits, c = "crimson", s = 3, label = "Event")

    name_o = os.path.join(root, f"O_{lc_name:05}_Q*.pt")    # Original
    
    list_ori = np.sort(glob.glob(name_o))                   # Listing the matching orignal files
    
    for file_ori in list_ori:

        out_path = f"{root}_plots"                      # Creating the plot output path
        strip_name = os.path.basename(file_ori)         # Stripping to the base name
        strip_name = os.path.splitext(strip_name)[0]    # Removing extension
        plot_name = strip_name.replace("O_", "")        # Removing the prefix for the plot
        aug_name = strip_name.replace("O_", "A_")       # Replacing the prefix for the augmented file

        ori = torch.load(file_ori)                      # Loading the original file
        time = torch.linspace(0, 1, len(ori["lc"]))     # Creating the time vector
        
        fig, ax = pl.subplot_mosaic([                   # Creating the figure
            ["ori"],
            ["aug"]
        ], figsize = (12, 12), layout = "constrained", sharex = True)

        _plot_ax(ax["ori"], time, ori)                  # Plotting the original data

        # We then check if the augmented data exists
        if file_aug := os.path.exists(os.path.join(root, aug_name)):
            aug = torch.load(file_aug)                  # If it does we load the file
            _plot_ax(ax["aug"], time, aug)              # And we plot it
        
        ax["ori"].set(ylabel = "Flux [original]")
        ax["ori"].legend(loc = "best")
        ax["aug"].set(ylabel = "Flux [augmented]")

        fig.suptitle(f"Sim {lc_name:05} :: Q{plot_name[-1]}")
        fig.supxlabel("Time")

        save_name = os.path.join(out_path, f"{plot_name}.png")
        fig.savefig(save_name)
        pl.close()

##################################################

def package(cfg: dict, augment: bool = False, filter_invalid: bool = True) -> None:
    """
    # The main caller to prepare the data

    Creates the list of files that requires processing and 
    """
    
    # Getting basic info
    path_data = os.path.join(cfg["path_data"], "*-*", "*_Q*.ftr")
    path_label = os.path.join(cfg["path_data"], "labels", "*-*", "*_Q*.pt")
    path_cp = os.path.join(cfg["path_data"], "config", "Complete_Parameters.ftr")
    path_out = cfg["path_prepared"]
    start_label = len(cfg["path_data"])

    ref = glob.glob(path_label)     # Listing all the labels available
    
    # We create the sim list based on the ref list to avoid length errors
    sim = []
    for ref_path in ref:

        base_path, label_name = os.path.split(ref_path)                         # Splitting the subpart of the path
        sim_base_path = base_path[:start_label] + base_path[start_label+7:]     # Removing the labels directory
        sim_name = label_name.replace("label", "sim").replace(".pt", ".ftr")    # Renaming the label in sim
        sim_path = os.path.join(sim_base_path, sim_name)                        # Creating the whole new path
        
        sim.append(sim_path)                                                    # Appending to the list

    ref = np.char.array(ref)
    sim = np.char.array(sim)

    # Reading the parameters file
    params = pd.read_feather(path_cp)

    # Preparing the iterator for the multi-process pool
    iterator = zip(
        iter(sim), iter(ref),
        [params] * len(sim), [path_out] * len(sim),
        [filter_invalid] * len(sim), [augment] * len(sim)
    )

    # Computing the number of CPU and starting the pool
    num_cpu = mp.cpu_count() - 1
    with mp.Pool(processes = num_cpu) as pool:
        pool.map(func = _wrapper_prepare_single, iterable = iterator)   # Calling the wrapper with the iterator

##########

def _wrapper_prepare_single(args) -> None:
    """
    # Wrapper for the multi-process of the preparation function
    """
    prepare_single(*args)

##########

def prepare_single(
        lc_name: str, ref_name: str,
        params: pd.DataFrame, path_out: str,
        filter_invalid: bool, augment: bool
    ) -> None:
    """
    # Packaging a simulation files into the .pt format for training.

    To simplify loading during training, we directly create a torch file with the lc and reference.

    We also have the possibility of augmenting the data by flipping the data time-wise. This is
    done by fitting a spline with a long fit window to only detrend from the long term effect.
    We output two files, the `O` and `A` version for the same light curve, original and augmented,
    respectively.
    """

    lc_base_name, ref_base_name = os.path.basename(lc_name), os.path.basename(ref_name)
    quarter = int(lc_base_name.split("_Q")[1][0])   # Extracting quarter
    print(PACKAGING, ORIGINAL, f":: {lc_base_name} - {ref_base_name}")

    light_curves = torch.from_numpy(                # Loading LC tensor
        pd.read_feather(lc_name).to_numpy(          # Casting as numpy
            dtype = np.float32,                     # We cast to float 32
            na_value = 0                            # Removing NaNs
        )
    )
    ref = torch.load(ref_name, weights_only = True)[:, cst.POINTS_KEEP]  # The ref file

    # We keep the same number of points as the old version of PlatoSim
    time = light_curves[cst.POINTS_KEEP, 0]         # Getting the time
    lc = light_curves[cst.POINTS_KEEP, -1]          # Getting the CoAdd

    if filter_invalid and any(lc.isinf()):          # If any inf is detected, we do not create the file
        print(FAILED, "inf", ORIGINAL, f":: {lc_base_name} - {ref_base_name}")
        return None
    elif filter_invalid and any(lc.isnan()):        # If any nan same
        print(FAILED, "nan", ORIGINAL, f":: {lc_base_name} - {ref_base_name}")
        return None
    
    # Getting the stellar parameters
    sim_index = int(os.path.basename(lc_name)[3:8])
    pos = params["index_sim"] == sim_index
    stellar = params[pos][cst.SIMS_PARAMS_PREP].reset_index(drop = True).T.to_dict("dict")[0]   # #ILovePython
    stellar["transit_snr"] = params[pos][f"transit_snr_{quarter}"].item()

    # The output object for orignal lc
    to_out = {
        "lc": lc,
        "ref": ref
    }
    to_out.update(stellar)  # And we add the stellar parameters

    # Naming the default file and writing it
    out_name = os.path.basename(lc_name).replace("sim", "O_").replace(".ftr", ".pt")
    out_name = os.path.join(path_out, out_name)
    write_file(to_out, out_name)

    # If we don't augment we skip to next iteration
    if not augment:
        return None
    
    print(PACKAGING, AUGMENTED, f":: {lc_base_name} - {ref_base_name}")
    continuum = ref[0]                      # Isolating the continuum

    # To circumvent needing a billion GiB of RAM, we use a shorter array
    pts = torch.arange(0, len(lc), 10, dtype = torch.long)
    short_time, short_lc, short_continuum = time[pts], lc[pts], continuum[pts]

    spline = interp.UnivariateSpline(
        x = short_time,                 # X data is time
        y = short_lc,                   # Y data is flux
        w = short_continuum,            # The continuum is weight
        k = ORDER_K,                    # Order of the spline
        s = SMOOTHING                   # Smoothing of the spline
    )

    # Computing the time-inverted curve
    trend = torch.from_numpy(spline(time))  # We compute the trend
    flattened = lc - trend                  # We detrend the current lc
    flipped = flattened.flip(0)             # Flipping time-wise
    final = flipped + trend                 # Re-adjusting the new curve
    ref_f = ref.flip(1)                     # Flipping the ref time-wise

    if filter_invalid and any(final.isinf()):      # If any inf is detected, we do not create the file
        print(FAILED, "inf", AUGMENTED, f":: {lc_base_name} - {ref_base_name}")
        return None
    elif filter_invalid and any(final.isnan()):    # If any nan same
        print(FAILED, "nan", AUGMENTED, f":: {lc_base_name} - {ref_base_name}")
        return None

    # The output object for augmented lc
    to_out = {
        "lc": final,
        "ref": ref_f
    }
    to_out.update(stellar)  # And we add the stellar parameters

    # Naming the augmented file and writing it
    out_name = os.path.basename(lc_name).replace("sim", "A_").replace(".ftr", ".pt")
    out_name = os.path.join(path_out, out_name)
    write_file(to_out, out_name)

########################################################################################################################

def write_file(obj, f_name: str) -> None:
    with open(f_name, "wb+") as f:
        torch.save(obj, f)

########################################################################################################################

def inspect(l: list[int], cfg: dict) -> None:

    root = cfg["path_prepared"]
    
    for elem in l:
        inspect_lc(root, elem)
