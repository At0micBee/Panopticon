"""
# Dataset for Plato Sim
"""

########################################################################################################################

# Model related modules
from .. import constants as cst         # Constant values

# Standard Python modules
import os                               # System operations
import glob                             # Unix style name parser
import numpy as np                      # Matrices

# Torch and its required subparts
import torch                            # Machine learning and tensors
from torch import Tensor                # The tensor type
from torch.utils.data import Dataset    # The dataset class

########################################################################################################################
# Creating the Dataset class to use for iteration

class LightCurveSet(Dataset):
    """
    # Dataset handling Plato light curves

    Using the prepared data from the pipeline, see the `readme` for more info.
    """

    def __init__(self, path: str, transform = None, target_transform = None) -> None:
        """
        # Dataset initialization

        We load the light curves and their labels. We load the entirety of the available data in a single dataset,
        splitting the data into `Subsets` later on. This simplifies the loading process and ensures reproducibility
        of each required `Subset`.

        - `path`: The path to the directory containing all the light curves to use.
        - `transform`: The transformation to apply on the input.
        - `target_transform`: The transformation to apply on the labels.
        """

        self.path = path                                                                # Storing the path
        self.path_files = os.path.join(self.path, "*_Q*.pt")                            # Path to files

        self.files = np.sort(glob.glob(self.path_files))                                # Constructing input file list

        self.transform = transform                                                      # Storing the transform
        self.target_transform = target_transform                                        # Storing the labels transform

    ############################################################

    @staticmethod
    def normalize(curve: Tensor) -> Tensor:
        """
        # Normalizing a curve

        Computes the normalization of an input vector in the range [0, 1], based on the min and max
        of the given vector. Note that the function also reshapes and casts the values to `f32`.

        - `curve`: the light curve to normalize, of shape [N].

        Returns a `Tensor` of shape [1, N], with values between 0 and 1.
        """

        clean = curve.reshape((1, -1))              # Re-shaping to [1, N]
        clean = clean.to(dtype = torch.float32)     # Casting to f32

        mini, maxi = clean.min(), clean.max()       # Getting min and max
        delta = maxi - mini                         # Getting delta

        return (clean - mini) / delta               # Normalizing curve
    
    @staticmethod
    def normalize_mean(curve: Tensor) -> Tensor:
        
        clean = curve.reshape((1, -1))              # Re-shaping to [1, N]
        clean = clean.to(dtype = torch.float32)     # Casting to f32

        mean = clean.mean()                         # Computing mean of the lc

        return (clean - mean) / mean                # Normalizing based on mean
    
    ############################################################

    @staticmethod
    def load_lc(lc_name: str) -> tuple[str, dict, Tensor, Tensor]:
        """
        # Loads a lightcurve and its metadata from a given path

        Given the path to a properly formatted data file, this function loads the lightcurve and its corresponding
        metadata. The data returned is processed (using `LightCurveSet.normalize`), and having a public method
        accessible from other modules ensures a single, unique, function to avoid inconsistencies or errors from
        other implementations.

        - `lc_name`: the path to the desired light curve file.

        Returns a tuple of data:
        - `lc_name`: the path of the processed data.
        - `sim_params`: the metadata of the simulation (planetary radius, stellar temperature, etc...).
        - `lc`: the normalized light curve.
        - `ref`: the reference for inference.
        """

        obj = torch.load(lc_name)                           # Loading the file
        lc, ref = obj["lc"], obj["ref"][1].unsqueeze(0)     # Getting the lc and ref

        lc = LightCurveSet.normalize(lc)                   # Normalizing

        # Getting the stellar params
        sim_params = { k: obj[k] for k in cst.SIMS_PARAMS }

        return lc_name, sim_params, lc, ref                 # Returning name, lc and ref
    
    ############################################################
    # Python built-in methods

    # Length of the dataset
    def __len__(self) -> int:
        return len(self.files)
    
    # Retrieving an item based on index
    def __getitem__(self, index: int) -> tuple[str, dict, Tensor, Tensor]:
        return LightCurveSet.load_lc( self.files[index] )

########################################################################################################################
