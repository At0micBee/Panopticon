"""
Invalid reference finder and cleaner.
"""

########################################################################################################################

# Model related modules
from src.model.dataset import LightCurveSet     # The dataset for the PLATO light curves
from src.utils import recovery                  # Transit recovery module

# Torch and its required subparts
from torch.utils.data import DataLoader         # Dataloader class and random split function

# Standard Python modules
import os                                       # System operations
import numpy as np                              # 
from termcolor import colored                   # Coloring terminal output

VALID = f"[{colored('Valid', color = 'green', attrs = ['bold'])}]"
INVALID = f"[{colored('Invalid', color = 'red', attrs = ['bold'])}]"
CUTOFF = f"[{colored('Cutoff', color = 'blue', attrs = ['bold'])}]"
DELETING = f"[{colored('Deleting', color = 'yellow', attrs = ['bold'])}]"

########################################################################################################################

def find_invalid(cfg: dict) -> None:
    """
    # Finding invalid files

    If no prediction boxes exist it is most likely an erroneous label, and we thus need to remove it from the 
    dataset. This functions checks the entire dataset for such files, and lists the invalid files for later deletion.

    Additionally, if we want to filter the sample in advance, we can pass the keyword and the bounds to remove
    the light curve from the dataset. Bounds are including the lower, and excluding the upper [min, max[.

    - cfg: the config file of the run
    """

    invalid_file = os.path.join(os.path.realpath(cfg["path_output"]), "invalid.txt")
    cutoff_file = os.path.join(os.path.realpath(cfg["path_output"]), "cutoff.txt")
    bounds_keys = cfg["bounds"]

    data = LightCurveSet(cfg["path_prepared"])
    loader = DataLoader(data)

    with open(invalid_file, "w+") as f:
        f.write(f"Invalid files in {cfg['path_prepared']}\n")
    
    with open(cutoff_file, "w+") as c:
        c.write(f"Removed from dataset due to bounds restrictions from {cfg['path_prepared']}\n")

    for f_name, lc_params, _, ref in loader:

        res = recovery.get_boxes(ref, 0.5)
        
        # Checking if the boxes exist
        if res is None:
            print(INVALID, f_name[0])

            with open(invalid_file, "a+") as f:
                f.write(f"{f_name[0]}\n")

        else:
            print(VALID, f_name[0])
        
        # We also run through the cutoff specified
        for key, (mi, ma) in bounds_keys.items():
            is_too_lo = lc_params[key] < mi         # Checking if under the min
            is_too_hi = lc_params[key] >= ma        # Checking if above the max

            if is_too_lo | is_too_hi:               # If either we flag it
                print(CUTOFF, f_name[0])

                with open(cutoff_file, "a+") as c:
                    c.write(f"{f_name[0]} :: param {key} = {lc_params[key].item()}, but cutoffs are [{mi}, {ma}[.\n" )

def clean_invalid(cfg: dict) -> None:

    invalid_path = os.path.join(os.path.realpath(cfg["path_output"]), "invalid.txt")
    cutoff_path = os.path.join(os.path.realpath(cfg["path_output"]), "cutoff.txt")

    invalid_list = np.genfromtxt(invalid_path, dtype = str, skip_header = 1)
    cutoff_list = np.genfromtxt(cutoff_path, dtype = str, delimiter = " :: ", skip_header = 1)[:,0]
    
    for file in invalid_list:
        print(DELETING, ":: invalid ::", file)
        os.remove(file)
    
    for file in cutoff_list:
        try:
            print(DELETING, ":: cutoff ::", file)
            os.remove(file)
            
        except OSError:         # Already deleted by the invalid detector
            print("    -> Deleted on the invalid pass.")

########################################################################################################################
