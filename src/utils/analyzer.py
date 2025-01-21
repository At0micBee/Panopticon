"""
# Analyzer

We load the output data of the model and plot relevant information to be visualized.
"""

########################################################################################################################

# Model related modules
from .. import constants as cst                 # Constant values

# Standard Python modules
import os                                       # System operations
import glob                                     # Unix style name parser
import pickle                                   # Binary file saver
import shutil                                   # OS utilities
import numpy as np                              # Numpy
import pandas as pd                             # Pandas
from numpy import ma                            # Masking tool
import colormaps as cmaps                       # Additional colormaps for pyplot
from termcolor import colored                   # Colored output
import matplotlib.pyplot as pl                  # Plotting module
import scipy.interpolate as interp              # Interpolation function
from matplotlib.ticker import MaxNLocator, PercentFormatter             # Ticks formatting
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   # Accessing cmaps list
from string import ascii_lowercase as alc       # Lower case ASCII

# Torch and its required subparts
import torch                                    # Machine learning and tensors
from torch import Tensor                        # The tensor type

# Updating default values in pyplot
pl.rcParams.update({
    "font.size": 12,                    # Better for paper
    "font.family": "serif",             # Better for paper
    "mathtext.fontset": "cm",           # Better font

    "lines.solid_capstyle": "round",    # Smoother
    "lines.dash_capstyle": "round",     # Smoother
    "hatch.linewidth": 0.3,             # Hatch line width, for pcolor

    "legend.edgecolor": "k"             # Darker edge color
})

########################################################################################################################
# Practical lambdas for data extractions

LOADING = f"[{colored('Loading', color = 'yellow', attrs = ['bold'])}]"
EXTRACTING = f"[{colored('Loading', color = 'blue', attrs = ['bold'])}]"
PLOTTING = f"[{colored('Plotting', color = 'green', attrs = ['bold'])}]"

CMAP = cmaps.guppy          # Loading the colormap
#CMAP_2D = cmaps.lavender    # Loading the colormap for 2D plots
CMAP_2D_2 = cmaps.dense     # Loading the colormap for secondary 2D plots
CMAP_2D_3 = cmaps.amp       # Loading the colormap for ternary 2D plots

thesis_color = {
    "RoyalPurple": "#5a349a",
    "Mulberry": "#a82596",
    "Emerald": "#00aaa0",
    "YellowGreen": "#8fd45a",
    "YellowOrange": "#f8980f",
    "OrangeRed": "#ed0e58"
}

cmap_cols = [thesis_color["Mulberry"], thesis_color["Emerald"], thesis_color["YellowGreen"]]

r = np.linspace(0, 1, len(cmap_cols), endpoint = True)
tuples = list(zip(r, cmap_cols))
CMAP_2D = LinearSegmentedColormap.from_list("", tuples)

custom_cols = [thesis_color["Mulberry"], thesis_color["Emerald"], thesis_color["YellowGreen"], thesis_color["YellowOrange"]]

r = np.linspace(0, 1, len(custom_cols), endpoint = True)
tuples = list(zip(r, custom_cols))
CMAP_CUSTOM = LinearSegmentedColormap.from_list("", tuples)

SAMPLE_KWARGS = {
    "color": "#f8980f",
    "edgecolor": "k",
    "linewidth": 0.1
}

METRICS_NAME = [
    "loss", "accuracy", "precision", "recall", "f1", "ap", "cm", "iou",
    "sample_radius_planet", "sample_transit_snr", "sample_transit_depth", "sample_transit_duration",
    "sample_transit_period", "sample_e", "sample_ip", "sample_radius_star", "sample_EB_period",
    "sample_EB_duration", "sample_BEB_period", "sample_BEB_duration", "sample_Teff", "sample_Prot",
    "sample_numax", "sample_logg",
    "found", "iou_cm"
]
"""
# Metrics to extract from the files
"""

TRAIN_METRICS = ["accuracy"]

########################################################################################################################
# Class for reduction

class Output:
    """
    # Output analysis class

    Loads a given output directory and the associated files. We compute the desired
    metrics for analysis and plot them for visualization.
    """

    def __init__(self, path: str) -> None:

        # Saving practical variables
        self.path = path                                                        # The current output
        self.path_process = os.path.join(self.path, "processed")                # Raw processed data path
        self.path_other_fmt = os.path.join(self.path_process, "other_fmt")      # Other fmt path
        self.path_files = os.path.join(self.path, "stats", "valid_*.pt")        # The validation files pattern
        self.path_files_train = os.path.join(self.path, "stats", "train_*.pt")  # The training files pattern
        self.sum_path = os.path.join(self.path, "processed", "summary.csv")     # Path to summary file
        self.rec_far_path = os.path.join(self.path, "processed", "rec_far.csv") # Path to summary file
        self.all_f_valid = sorted(glob.iglob(self.path_files))                  # The list of validation files
        self.all_f_train = sorted(glob.iglob(self.path_files_train))            # The list of training files

        # Loading all files
        self.device = torch.device("cpu")                               # Ensuring the processing in on CPU
        print(LOADING, f"Validation data...")
        self.loaded_f_valid = [torch.load(f, map_location = self.device) for f in self.all_f_valid]
        print(LOADING, f"Training data...")
        self.loaded_f_train = [torch.load(f, map_location = self.device) for f in self.all_f_train]

        self.thr_0 = cst.THRESHOLDS[0].item()                           # The zeroth entry in threshold

        if os.path.exists(self.path_process):                           # Check if the dir exists
            shutil.rmtree(self.path_process)                            # We remove
        os.mkdir(self.path_process)                                     # We re-create it
        os.mkdir(self.path_other_fmt)                                   # Creating the sub dir for other fmt

        self.stats = {}
        self.stats_train = {}

        ##################################################
        # Loading the validation data
        ##################################################

        for thr in cst.THRESHOLDS:

            thr = thr.item()

            # Extracting the data from the files
            for idx, loaded in enumerate(self.loaded_f_valid):
                print(EXTRACTING, f"Validation data for threshold {thr:.3} :: file {idx}")
                
                # If we are on the first iter, we create the basic tensor
                if idx == 0:
                    self.stats[thr] = {n: self._tensor_from_dict_val(n, loaded, thr) for n in METRICS_NAME}

                # If not, we stack
                else:
                    for n, s in self.stats[thr].items():
                        self.stats[thr][n] = torch.vstack(( s, self._tensor_from_dict_val(n, loaded, thr) ))
        
        ##################################################
        # Loading the training data
        ##################################################

        # Extracting the data from the files
        for idx, loaded in enumerate(self.loaded_f_train):
            print(EXTRACTING, f"Training data :: file {idx}")
            
            # If we are on the first iter, we create the basic tensor
            if idx == 0:
                self.stats_train = {n: self._tensor_from_dict_val(n, loaded, None) for n in TRAIN_METRICS}

            # If not, we stack
            else:
                for n, s in self.stats_train.items():
                    self.stats_train[n] = torch.vstack(( s, self._tensor_from_dict_val(n, loaded, None) ))
        
        # Epoch counter and running export creation
        self.epochs = torch.arange(0, len(self.stats[self.thr_0]["loss"])) + 1
        
        self.df_export = pd.DataFrame()
        self.df_rec_far = pd.DataFrame()
        
    ################################################################################

    @staticmethod
    def _tensor_from_dict_val(name: str, d: dict, thr: float) -> Tensor:

        if thr is not None:
            res = torch.stack( [v[thr][name] for v in d.values()] ).unsqueeze(dim = 0)
        
        else:
            res = torch.stack( [v[name] for v in d.values()] ).unsqueeze(dim = 0)
        
        return res

    ################################################################################

    def process(self) -> None:
        """
        # Processes the output in its entirety

        We compute the metrics here, and use them to evaluate the model. We also
        call the related plotting function for visualization.
        """

        self.plot_loss()
        self.plot_accuracy()
        self.plot_prec_recall()
        self.plot_f1()
        self.plot_ap()
        self.plot_iou()
        self.plot_matrix()
        self.threshold_values()
        self.plot_incorrect()
        self.plot_recovery_1d()
        self.plot_recovery_2d()
        self.plot_recovery()
        self.plot_custom()
        #self.plot_stellar_recovery_1d()
        #self.plot_stellar_recovery_2d()
        #self.report()

        self.df_export.to_csv(self.sum_path, index_label = "Epoch")
        self.df_rec_far.to_csv(self.rec_far_path, index_label = "Epoch")
    
    ################################################################################
    
    def save_fig(self, fig, name: str) -> None:
        """
        # Saves a fig to all correct format
        """
        
        fig_name = os.path.join(self.path_process, f"{name}.pdf")       # Path for pdf
        fig.savefig(fig_name)                                           # Saving

        fig_name = os.path.join(self.path_other_fmt, f"{name}.png")     # Path for png
        fig.savefig(fig_name, dpi = 600)                                # Saving
        
        fig_name = os.path.join(self.path_other_fmt, f"{name}.pkl")     # Path for pickle
        with open(fig_name, "wb+") as f_pkl:                            # Opening in binary mode
            pickle.dump(fig, f_pkl)                                     # Dumping the pickled data

    ################################################################################
            
    def report(self) -> None:

        print(PLOTTING, "Report")

        fig, ax = pl.subplot_mosaic([
            ["loss", "accuracy"],
            ["precision", "recall"],
            ["f1", "ap"],
            ["iou", "iou"]
        ], figsize = (8.27, 11.69), layout = "constrained")

        epoch_mean = torch.mean(self.stats[self.thr_0]["loss"], dim = 1)                    # Loss
        epoch_accuracy_train_mean = torch.mean(self.stats_train["accuracy"], dim = (1, 2))  # Accuracy, /!\ DURING TRAINING /!\

        ax["loss"].plot(self.epochs, epoch_mean, c = "k")
        ax["accuracy"].plot(self.epochs, epoch_accuracy_train_mean, c = "k", lw = 2, dashes = [4, 3])

        for thr in cst.THRESHOLDS[7:12]:

            thr = thr.item()
            col = CMAP(thr)

            epoch_accuracy_mean = torch.mean(self.stats[thr]["accuracy"], dim = (1, 2))
            epoch_precision_mean = torch.mean(self.stats[thr]["precision"], dim = (1, 2))
            epoch_recall_mean = torch.mean(self.stats[thr]["recall"], dim = (1, 2))
            epoch_f1_mean = torch.mean(self.stats[thr]["f1"], dim = (1, 2))
            epoch_ap_mean = torch.mean(self.stats[thr]["ap"], dim = 1)
            epoch_iou_mean = torch.mean(self.stats[thr]["iou"], dim = 1)

            ax["accuracy"].plot(self.epochs, epoch_accuracy_mean, c = col, alpha = 0.4, lw = 0.75, label = f"{thr:.2}")
            ax["precision"].plot(self.epochs, epoch_precision_mean, c = col, lw = 0.75, label = f"{thr:.2}")
            ax["recall"].plot(self.epochs, epoch_recall_mean, c = col, lw = 0.75, label = f"{thr:.2}")
            ax["f1"].plot(self.epochs, epoch_f1_mean, c = col, lw = 0.75, label = f"{thr:.2}")
            ax["ap"].plot(self.epochs, epoch_ap_mean, c = col, lw = 0.75, label = f"{thr:.2}")
            ax["iou"].plot(self.epochs, epoch_iou_mean, c = col, lw = 0.75, label = f"{thr:.2}")
        
        ax["loss"].set(ylabel = "Average loss")
        ax["recall"].sharey(ax["precision"])

        self.save_fig(fig, "report")
        pl.close()
    
    ################################################################################

    def plot_custom(self) -> None:
        """
        # Plotting a custom graph
        """
        print(PLOTTING, "Custom")

        epoch_mean = torch.mean(self.stats[self.thr_0]["loss"], dim = 1)

        fig, ax = pl.subplot_mosaic([
            ["loss", "iou"]
        ], layout = "constrained", figsize = (6.4*2/1.3, 4.8/1.3), sharex = True)

        ax["loss"].plot(self.epochs, epoch_mean, color = "black")

        for thr in cst.THRESHOLDS[1::2]:
            thr = thr.item()

            epoch_mean = torch.mean(self.stats[thr]["iou"], dim = 1)

            self.df_export[f"IOU {thr:.2}"] = epoch_mean

            col = CMAP_CUSTOM(thr)
            ax["iou"].plot(self.epochs, epoch_mean, color = col, label = f"{thr:.2}", linewidth = 0.75)

        ax["iou"].yaxis.set_major_formatter(PercentFormatter(xmax = 1, decimals = 0))

        fig.supxlabel("Epoch")
        ax["iou"].legend(loc = "lower right")
        ax["iou"].set(ylabel = "IOU score")
        ax["loss"].set(ylabel = "Average loss")

        ax["iou"].legend(bbox_to_anchor = (0.96, 0.5), loc = "center left", fontsize = 9, framealpha = 1, edgecolor = "black")

        self.save_fig(fig, "iou_loss")
        pl.close()

    def plot_loss(self) -> None:
        """
        # Plotting the loss over time
        """

        print(PLOTTING, "Loss")

        # Computing mean and standard deviation
        epoch_mean = torch.mean(self.stats[self.thr_0]["loss"], dim = 1)

        fig = pl.figure(layout = "constrained")

        pl.plot(self.epochs, epoch_mean, color = "black")

        pl.xlabel("Epoch")
        pl.ylabel("Average loss")

        self.save_fig(fig, "loss")
        pl.close()

        # Updating the running export
        self.df_export["Loss"] = epoch_mean
    
    ################################################################################

    def plot_accuracy(self) -> None:

        print(PLOTTING, "Accuracy")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")

        # Computing the mean of the training accuracy
        epoch_accuracy_train_mean = torch.mean(self.stats_train["accuracy"], dim = (1, 2))

        ax.plot(self.epochs, epoch_accuracy_train_mean, color = "black", label = "Train", dashes = [4, 3], linewidth = 2)

        for thr in cst.THRESHOLDS[7:12]:
            thr = thr.item()

            epoch_accuracy_mean = torch.mean(self.stats[thr]["accuracy"], dim = (1, 2))

            col = CMAP(thr)
            ax.plot(self.epochs, epoch_accuracy_mean, color = col, label = f"{thr:.2}", linewidth = 0.75)

        pl.xlabel("Epoch")
        pl.ylabel("Accuracy")
        ax.legend(bbox_to_anchor = (0.96, 0.5), loc = "center left", fontsize = 9, framealpha = 1, edgecolor = "black")

        self.save_fig(fig, "accuracy")
        pl.close()

    ################################################################################

    def plot_prec_recall(self) -> None:

        print(PLOTTING, "Precision and Recall")

        fig, ax = pl.subplot_mosaic([
            ["precision", "recall"]
        ], figsize = (6.4*2 / 1.3, 4.8 / 1.3), layout = "constrained", sharex = True)
        ax["recall"].yaxis.tick_right()
        ax["recall"].yaxis.set_label_position("right")

        for thr in cst.THRESHOLDS[1::2]:
            thr = thr.item()

            epoch_precision_mean = torch.mean(self.stats[thr]["precision"], dim = (1, 2))
            epoch_recall_mean = torch.mean(self.stats[thr]["recall"], dim = (1, 2))

            self.df_export[f"Precision {thr:.2}"] = epoch_precision_mean
            self.df_export[f"Recall {thr:.2}"] = epoch_recall_mean
        
            col = CMAP_CUSTOM(thr)
            ax["precision"].plot(self.epochs, epoch_precision_mean, color = col, label = f"{thr:.2}", linewidth = 0.75)
            ax["recall"].plot(self.epochs, epoch_recall_mean, color = col, label = f"{thr:.2}", linewidth = 0.75)
        
        ax["precision"].legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = 7.8, framealpha = 1, edgecolor = "black")
        ax["precision"].set(ylabel = "Precision")
        ax["recall"].set(ylabel = "Recall")
        ax["precision"].yaxis.set_major_formatter(PercentFormatter(xmax = 1, decimals = 0))
        ax["recall"].yaxis.set_major_formatter(PercentFormatter(xmax = 1, decimals = 0))
        fig.supxlabel("Epoch")

        self.save_fig(fig, "metrics")
        pl.close()
    
    ################################################################################

    def plot_f1(self) -> None:
        """
        # Plotting the f1 score
        """

        print(PLOTTING, "F1 score")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")

        for thr in cst.THRESHOLDS:
            thr = thr.item()

            epoch_f1_mean = torch.mean(self.stats[thr]["f1"], dim = (1, 2))

            self.df_export[f"F1 {thr:.2}"] = epoch_f1_mean

            col = CMAP(thr)
            ax.plot(self.epochs, epoch_f1_mean, color = col, label = f"{thr:.2}", linewidth = 0.75)

        pl.legend(loc = "lower right")
        pl.xlabel("Epoch")
        pl.ylabel(r"$F_1$ score")
        ax.legend(bbox_to_anchor = (0.96, 0.5), loc = "center left", fontsize = 9, framealpha = 1, edgecolor = "black")

        self.save_fig(fig, "f1")
        pl.close()
    
    ################################################################################

    def plot_ap(self) -> None:
        """
        # Plotting the average precision
        """

        print(PLOTTING, "Average Precision")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")
        
        for thr in cst.THRESHOLDS:
            thr = thr.item()

            epoch_ap_mean = torch.mean(self.stats[thr]["ap"], dim = 1)

            self.df_export[f"AP {thr:.2}"] = epoch_ap_mean

            col = CMAP(thr)
            ax.plot(self.epochs, epoch_ap_mean, color = col, label = f"{thr:.2}", linewidth = 0.75)

        pl.xlabel("Epoch")
        pl.ylabel("AP score")
        ax.legend(bbox_to_anchor = (0.96, 0.5), loc = "center left", fontsize = 9, framealpha = 1, edgecolor = "black")

        self.save_fig(fig, "ap")
        pl.close()
    
    ################################################################################

    def plot_iou(self) -> None:
        """
        # Plotting the jaccard score over time
        """

        print(PLOTTING, "IOU")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")

        for thr in cst.THRESHOLDS:
            thr = thr.item()

            epoch_mean = torch.mean(self.stats[thr]["iou"], dim = 1)

            self.df_export[f"IOU {thr:.2}"] = epoch_mean

            col = CMAP(thr)
            pl.plot(self.epochs, epoch_mean, color = col, label = f"{thr:.2}", linewidth = 0.75)

        pl.legend(loc = "lower right")
        pl.xlabel("Epoch")
        pl.ylabel("IOU score")
        ax.legend(bbox_to_anchor = (0.96, 0.5), loc = "center left", fontsize = 9, framealpha = 1, edgecolor = "black")

        self.save_fig(fig, "iou")
        pl.close()

    ################################################################################
    
    def plot_matrix(self) -> None:
        """
        # Plotting the confusion matrix
        """

        print(PLOTTING, "Confusion Matrix (point-wise)")

        fig, ax = pl.subplot_mosaic([
            ["TP", "FN"],
            ["FP", "TN"]
        ], figsize = (8, 6), layout = "constrained", sharex = True)

        ax["FN"].yaxis.tick_right()
        ax["TN"].yaxis.tick_right()
        ax["FN"].yaxis.set_label_position("right")
        ax["TN"].yaxis.set_label_position("right")

        lines, labels = [], []

        for thr in cst.THRESHOLDS:

            thr = thr.item()
            col = CMAP(thr)

            # Shape [epoch, num of batches, num_label, 2, 2], 2x2 from confusion matrix
            count_per_class = self.stats[thr]["cm"].sum(dim = 1)

            ax["TP"].plot(self.epochs, count_per_class[:, 1, 1], color = col, label = f"{thr:.2}", linewidth = 0.75)
            ax["FP"].plot(self.epochs, count_per_class[:, 0, 1], color = col, label = f"{thr:.2}", linewidth = 0.75)
            ax["TN"].plot(self.epochs, count_per_class[:, 0, 0], color = col, label = f"{thr:.2}", linewidth = 0.75)
            l = ax["FN"].plot(self.epochs, count_per_class[:, 1, 0], color = col, label = f"{thr:.2}", linewidth = 0.75)
            lines.append(l)
            labels.append(f"{thr:.2}")

        fig.supxlabel("Epoch")

        for ax_name, a in ax.items():
            a.ticklabel_format(useMathText = True)
            a.set_ylabel(ax_name)

        self.save_fig(fig, "matrix")
        pl.close()

    ################################################################################

    def threshold_values(self) -> None:

        computed = {
            "iou": torch.stack([self.stats[thr.item()]["iou"].mean(dim = 1) for thr in cst.THRESHOLDS]),
            "f1": torch.stack([self.stats[thr.item()]["f1"].mean(dim = (1, 2)) for thr in cst.THRESHOLDS]),
            "ap": torch.stack([self.stats[thr.item()]["ap"].mean(dim = 1) for thr in cst.THRESHOLDS]),
            "pre": torch.stack([self.stats[thr.item()]["precision"].mean(dim = (1, 2)) for thr in cst.THRESHOLDS]),
            "rec": torch.stack([self.stats[thr.item()]["recall"].mean(dim = (1, 2)) for thr in cst.THRESHOLDS])
        }
        
        for metric in computed.keys():

            print(PLOTTING, f"Threshold for {metric}")

            fig = pl.figure(layout = "constrained")

            data = computed[metric].T

            for idx, epoch in enumerate(data):

                if idx < len(data) - 1:
                    pl.plot(cst.THRESHOLDS, epoch, color = "black", alpha = idx / len(data), linewidth = 0.6)
                else:
                    pl.plot(cst.THRESHOLDS, epoch, color = "crimson", alpha = idx / len(data), label = f"Last epoch")

            pl.legend(loc = "lower center")
            pl.xlabel("Threshold value")
            pl.ylabel("Score")

            self.save_fig(fig, f"thresholds_{metric}")
            pl.close()

        print(PLOTTING, "Threshold for last epoch")

        # Comparing all values at the last epoch
        fig = pl.figure(layout = "constrained")

        pl.plot(cst.THRESHOLDS, computed["iou"].T[-1], color = "crimson", label = "IOU")
        pl.plot(cst.THRESHOLDS, computed["f1"].T[-1], color = "teal", label = "F1 score")
        pl.plot(cst.THRESHOLDS, computed["ap"].T[-1], color = "purple", label = "Average Precision")
        pl.plot(cst.THRESHOLDS, computed["pre"].T[-1], color = "goldenrod", label = "Precision")
        pl.plot(cst.THRESHOLDS, computed["rec"].T[-1], color = "darkorchid", label = "Recall")

        # Creating a refined grid and fitting spline
        large_t = torch.linspace(0.05, 0.95, 1000)

        spline = interp.splrep(
            x = cst.THRESHOLDS,                                 # X data is thresholds
            y = computed["iou"].T[-1],                          # Y data is the iou
            k = 4                                               # Order of the spline
        )

        fit_par = interp.splev(large_t, spline)                 # Fitting the spline to an HD threshold array
        max_arg = fit_par.argmax()                              # Getting the max arg of the result
        max_thr, max_fit = fit_par[max_arg], large_t[max_arg]   # Getting the max threshold and IOU score

        # Computing the index of the best thr
        self.best_arg_thr = torch.argmin(torch.abs(cst.THRESHOLDS - max_fit)).item()

        pl.plot(
            large_t, fit_par,
            color = "black", alpha = 0.5, zorder = 0,
            label = f"IOU spline fit (max: {max_fit:.5})"
        )

        pl.axvline(max_fit.item(), color = "black", alpha = 0.8, dashes = [1, 3, 5, 3])

        pl.xlabel("Threshold value")
        pl.ylabel("Corresponding score")
        pl.legend(loc = "best")

        self.save_fig(fig, "thresholds_last")
        pl.close()

    ################################################################################

    def plot_incorrect(self) -> None:

        print(PLOTTING, "Confusion Matrix (event-wise)")

        self.fap_thr = {}
        
        fig, ax = pl.subplot_mosaic([
            ["Missed", "Recovered", "False Positive"]
        ], figsize = (12, 4.8), layout = "constrained")

        for thr in cst.THRESHOLDS:
            thr = thr.item()

            iou_cm = self.stats[thr]["iou_cm"].sum(dim = (1, 2))

            #self.fap_epoch = iou_cm[:,2] / total_count  # FAP at each epoch
            self.fap_thr[thr] = (100 * iou_cm[:,2] / (iou_cm[:,1] + iou_cm[:,2])).numpy()#.round(2)

            col = CMAP(thr)
            ax["Missed"].plot(self.epochs, iou_cm[:,0], color = col, label = f"{thr:.2}", linewidth = 0.75)
            ax["Recovered"].plot(self.epochs, iou_cm[:,1], color = col, label = f"{thr:.2}", linewidth = 0.75)
            ax["False Positive"].plot(self.epochs, iou_cm[:,2], color = col, label = f"{thr:.2}", linewidth = 0.75)

        for ax_name, a in ax.items():
            a.set_title(ax_name)
        ax["False Positive"].legend(bbox_to_anchor = (1, 0.5), loc = "center left", fontsize = 7.8, framealpha = 1, edgecolor = "black")

        fig.supxlabel("Epoch")
        fig.supylabel("Occurrences")

        self.save_fig(fig, "cm_transits")
        pl.close()
    
    ################################################################################

    def plot_recovery_1d(self) -> None:

        print(PLOTTING, "Planetary recovery 1D")

        self.best_thr = cst.THRESHOLDS[self.best_arg_thr].item()     # Best arg computed in threshold_values()

        if False:
            print("Manually setting threshold.")
            self.best_thr = cst.THRESHOLDS[6].item()
        
        print("Using threshold:", self.best_thr)

        # Defining the bins for each parameter
        self.edges = {
            "radius_planet": torch.linspace(0, 16.5, 25), #  max used to be 20.5
            "transit_snr": torch.arange(0, 100, 3, dtype = torch.float32),    # 5421 vs 100
            "transit_depth": torch.linspace(50, 900, 25),
            #"transit_duration": torch.linspace(0, 1.5, 25),
            #"transit_period": torch.linspace(0, 370, 25),
            #"e": torch.linspace(0, 0.15, 25),
            #"ip": torch.linspace(85, 90, 25),
            #"radius_star": torch.linspace(0.5, 2.5, 25),
            "Teff": torch.linspace(4730, 6554, 25),
            "Prot": torch.linspace(2, 64, 25),
            "numax": torch.linspace(190, 6510, 25),
            "logg": torch.linspace(3.7, 4.8, 25)
        }

        self.x_labels = {
            "radius_planet": "Planetary radius [$R_\oplus$]",
            "transit_snr": "Transit SNR",
            "transit_depth": "Transit depth [ppm]",
            "transit_duration": "Transit duration $T_{14}$ [days]",
            "transit_period": "Orbital period [days]",
            "e": "Eccentricity",
            "ip": "Inclination [deg]",
            "radius_star": "Stellar radius [$R_\odot$]",
            "Teff": "$T_\mathrm{eff}$ [K]",
            "Prot": "$P_\mathrm{rot}$ [days]",
            "numax": r"$\nu_\mathrm{max}$",
            "logg": "$\log(g)$"
        }

        # Computing the samples and recovered
        self.samples = { param: self.stats[self.thr_0][f"sample_{param}"].flatten(start_dim = 1)[0] for param in self.edges.keys() }
        self.samples["transit_depth"] *= 1e6                        # Getting depth in ppm
        self.is_planet = ~self.samples["radius_planet"].isnan()     # Where no planet -> NaN

        #print(max(self.samples["transit_snr"][self.is_planet]), min(self.samples["transit_snr"][self.is_planet]))

        # Getting the recovered positions, last epoch
        self.recovered = self.stats[self.best_thr]["found"].flatten(start_dim = 1)[-1]  # Where the recovery happened
        self.is_recovered_planet = self.is_planet & self.recovered                      # Where the recovery is a planet
        self.is_recovered_failed = self.is_planet & ~self.recovered                     # Where the planets aren't recovered

        fig, ax = pl.subplots(nrows = len(self.edges), ncols = 1, figsize = (6.4, 3.2*len(self.edges)), layout = "constrained")

        for (row, (param, sample)) in enumerate(self.samples.items()):

            sample_hist, _, bars = ax[row].hist(
                x = sample[self.is_planet],
                bins = self.edges[param],
                color = thesis_color["RoyalPurple"],# "mediumpurple", #"purple",
                label = "Real population" if row == 0 else None,
                edgecolor = "k",
                linewidth = 0.1
            )
            recovered_hist, _, _ = ax[row].hist(
                x = self.samples[param][self.is_recovered_planet],
                bins = self.edges[param],
                color = thesis_color["YellowGreen"],#"#70db93",# "mediumseagreen",
                label = "Recovered" if row == 0 else None,
                edgecolor = "k",
                linewidth = 0.1
            )

            # Computing ratio, masking where no data
            recovered_frac = (100 * np.divide(recovered_hist, sample_hist, where = sample_hist != 0)).round(1)
            recovered_frac = ma.masked_where(sample_hist == 0, recovered_frac)

            ax_rec = ax[row].twinx()
            offset = (self.edges[param][1] - self.edges[param][0]) / 2
            ax_rec.plot(self.edges[param][:-1] + offset, recovered_frac, c = "#f8980f", marker = "h", mfc = "w", mec = "#f8980f", ms = 7)
            ax_rec.yaxis.set_major_formatter(PercentFormatter())

            # Showing key info depending on the plot
            match param:
                case "transit_snr":
                    ax[row].arrow(
                        85, 27, 12, 0, width = 0.5,
                        head_width = 1, length_includes_head = True, overhang = 0.1,
                        ec = "k", fc = "w"
                    )
                    ax[row].text(
                        91, 26, "Axis\ntruncated",
                        va = "top", ha = "center", fontsize = 10
                    )
                
                case _:
                    if row == 0:
                        #ax[row].legend(loc = "center right", markerfirst = False, edgecolor = "k")
                        pass
                
            """ case "radii":
                ax[row].axvline(1, label = "Earth", color = "k", linewidth = 0.75, alpha = 0.5)
                ax[row].axvline(3.860397297, label = "Neptune", color = "k", linewidth = 0.75, dashes = [7, 4], alpha = 0.5)
                ax[row].axvline(11.208980731, label = "Jupiter", color = "k", linewidth = 0.75, dashes = [3, 3, 10, 3], alpha = 0.5)
                ax[row].legend(loc = "upper right", markerfirst = False, edgecolor = "k")

            case "transit_period":
                ax[row].axvline(cst.Q_LENGTH / cst.SEC_DAY, label = "Quarter length", color = "k", linewidth = 0.75)
                ax[row].legend(loc = "lower left", markerfirst = False, edgecolor = "k")
            
            case "transit_depth":
                ax[row].axvline(84, label = "Earth depth", color = "k", linewidth = 0.75)
                ax[row].legend(loc = "center right", markerfirst = False, edgecolor = "k") """

            #ax[row].bar_label(bars, fontsize = 10, rotation = 90, labels = recovered_frac, fmt = "{}%")
            ax[row].set(
                xlim = (self.edges[param][0], self.edges[param][-1]),
                ylim = (0, max(sample_hist) * 1.12),
                xlabel = self.x_labels[param]
            )
            #ax[row].yaxis.set_label_position("right")
            #ax[row].set_ylabel(self.x_labels[param], rotation = 270, labelpad = 13)
            ax[row].text(1e-3, 1-8e-3, f"({alc[row]})", transform = ax[row].transAxes, ha = "left", va = "top")

        # General
        #fig.supylabel("# of occurrences")

        self.save_fig(fig, "recovery")
        pl.close()
    
    def plot_stellar_recovery_1d(self) -> None:

        print(PLOTTING, "Stellar recovery 1D")

        # Defining the bins for each parameter
        self.edges_stellar = {
            "EB_period": torch.linspace(0, 365, 25),
            #"EB_duration": torch.linspace(0, 1.3, 25)
        }

        self.x_labels_stellar = {
            "EB_period": "Period [day]",
            #"EB_duration": "Duration [day]"
        }
        
        self.samples_stellar_ho = { param: self.stats[self.thr_0][f"sample_{param}"].flatten(start_dim = 1)[0] for param in self.edges_stellar.keys() }     # Host
        self.samples_stellar_bg = { param: self.stats[self.thr_0][f"sample_B{param}"].flatten(start_dim = 1)[0] for param in self.edges_stellar.keys() }    # Background

        # Merging the two populations
        self.samples_stellar = {}
        for param in self.edges_stellar.keys():
            host = self.samples_stellar_ho[param]
            bg = self.samples_stellar_bg[param]

            bg_exists = ~bg.isnan()
            host[bg_exists] = bg[bg_exists]

            self.samples_stellar[param] = host
        
        # TODO: don't forget to remove comment when adding the duration
        #self.samples_stellar["EB_duration"] /= cst.SEC_DAY
        
        self.is_binary = ~self.samples_stellar["EB_period"].isnan()                         # Where no binary -> NaN
        self.recovered_bin = self.stats[self.best_thr]["found"].flatten(start_dim = 1)[-1]  # Where the recovery happened
        self.is_recovered_binary = self.is_binary & self.recovered_bin                      # Where the recovery is a binary
        self.is_recovered_binary_f = self.is_binary & ~self.recovered_bin                   # Where the binary aren't recovered

        fig, ax = pl.subplots(nrows = len(self.edges_stellar), ncols = 1, figsize = (6.4, 4.8*len(self.edges_stellar)), layout = "constrained", squeeze = False)

        for (row, (param, sample)) in enumerate(self.samples_stellar.items()):

            sample_hist, _, bars = ax[0][row].hist(
                x = sample[self.is_binary],
                bins = self.edges_stellar[param],
                color = thesis_color["RoyalPurple"],
                label = "Actual",
                edgecolor = "k",
                linewidth = 0.1
            )
            recovered_hist, _, _ = ax[0][row].hist(
                x = self.samples_stellar[param][self.is_recovered_binary],
                bins = self.edges_stellar[param],
                color = thesis_color["YellowGreen"],
                label = "Recovered",
                edgecolor = "k",
                linewidth = 0.1
            )

            print(recovered_hist.sum(), sample_hist.sum(), recovered_hist.sum() / sample_hist.sum())

            # Computing ratio, masking where no data
            recovered_frac = (100 * np.divide(recovered_hist, sample_hist, where = sample_hist != 0)).round(1)
            recovered_frac = ma.masked_where(sample_hist == 0, recovered_frac)

            ax_rec = ax[0][row].twinx()
            offset = (self.edges_stellar[param][1] - self.edges_stellar[param][0]) / 2
            ax_rec.plot(self.edges_stellar[param][:-1] + offset, recovered_frac, c = "#f8980f", marker = "h", mfc = "w", mec = "#f8980f", ms = 7)
            ax_rec.yaxis.set_major_formatter(PercentFormatter(decimals = 1))

            ax[0][row].set_xlabel(self.x_labels_stellar[param])
            #ax[row].bar_label(bars, fontsize = 6, rotation = 90, labels = recovered_frac)

        # General
        fig.supylabel("# of occurrences")

        self.save_fig(fig, "recovery_binaries")
        pl.close()

    def plot_recovery_2d(self) -> None:

        print(PLOTTING, "Planetary recovery 2D")

        #ls = np.array([*self.edges.keys()]) # Getting the names of the parameters to mesh
        ls = np.array(["transit_snr", "Teff", "Prot", "logg", "numax"])
        rows, cols = np.meshgrid(ls, ls)    # Creating the rows and cols names for corner

        # Recovered fraction and original distribution
        fig, ax = pl.subplots(
            nrows = rows.shape[0], ncols = cols.shape[1],
            sharex = "col", sharey = "row",
            layout = "constrained", figsize = (8, 7)
        )

        # Failed and original distribution
        fig_f, ax_f = pl.subplots(
            nrows = rows.shape[0], ncols = cols.shape[1],
            sharex = "col", sharey = "row",
            layout = "constrained", figsize = (6.4 * 3, 5.5 * 3)
        )

        # Going through each of the subplots
        for row, col in np.ndindex(rows.shape):
            
            # Diagonal -> 1d distribution
            if row == col:
                #ax[row, col].get_shared_y_axes().remove(ax[row, col])        # Decoupling the ax from the shared y axis
                ax[-1, col].set_xlabel(self.x_labels[rows[row, col]])
                ax[row, 0].set_ylabel(self.x_labels[cols[row, col]])
                #ax_f[row, col].get_shared_y_axes().remove(ax[row, col])      # Decoupling the ax from the shared y axis
                ax_f[-1, col].set_xlabel(self.x_labels[rows[row, col]])
                ax_f[row, 0].set_ylabel(self.x_labels[cols[row, col]])

                param = rows[row, col]
                sample = self.samples[param]

                # Possibility to pass the count on the right ax
                right_ax = ax[row, col].twinx()
                right_ax_f = ax_f[row, col].twinx()

                actual, _, _ = right_ax.hist(x = sample[self.is_planet], bins = self.edges[param], color = thesis_color["RoyalPurple"], label = "Actual")
                _, _, _ = right_ax.hist(
                    x = sample[self.is_recovered_planet],
                    bins = self.edges[param],
                    color = thesis_color["YellowGreen"],
                    label = "Recovered"
                )

                failed, _, _ = right_ax_f.hist(x = sample[self.is_recovered_failed], bins = self.edges[param], color = CMAP_2D_2(0.8), label = "Failed")

                right_ax.tick_params(axis = "y", which = "both", right = False, labelright = False)
                right_ax_f.tick_params(axis = "y", which = "both", right = False, labelright = False)

                """ if row == 0:
                    right_ax.legend(loc = "upper right")
                    right_ax_f.legend(loc = "upper right") """
            
            # Lower left corner -> 2d fraction recovered
            elif row > col:
                param_row, param_col = rows[row, col], cols[row, col]                                       # Parameter names
                edges = (self.edges[param_row], self.edges[param_col])                                      # Edges of the bins
                mesh_edge = np.meshgrid(*edges)

                sample_row = self.samples[param_row][self.is_planet].numpy()                                # Sample row
                sample_col = self.samples[param_col][self.is_planet].numpy()                                # Sample col

                recovered_row = self.samples[param_row][self.is_recovered_planet].numpy()                   # Recovered row
                recovered_col = self.samples[param_col][self.is_recovered_planet].numpy()                   # Recovered col

                failed_row = self.samples[param_row][self.is_recovered_failed].numpy()                      # Failed row
                failed_col = self.samples[param_col][self.is_recovered_failed].numpy()                      # Failed cow

                baseline, _, _ = np.histogram2d(sample_row, sample_col, bins = edges)                       # Histogram of samples
                recovered, _, _ = np.histogram2d(recovered_row, recovered_col, bins = edges)                # Histogram of recovery
                recovered_frac = (100 * np.divide(recovered, baseline, where = baseline != 0)).round(1)     # Fraction recovered / samples
                recovered_frac = ma.masked_where(baseline == 0, recovered_frac)                             # Masking where no data
                recovered_frac = recovered_frac.T                                                           # Transpose for visual accuracy

                mesh = ax[row, col].pcolormesh(                                                             # Plotting the recovery rate
                    *mesh_edge, recovered_frac,                                                             # Axes and data
                    cmap = CMAP_2D, edgecolor = "face",                                                     # Color and filling mode between pixels
                    vmin = 0, vmax = 100                                                                    # Ensuring min and max between 0% and 100%
                )

                failed_base, _, _ = np.histogram2d(failed_row, failed_col, bins = edges)                    # Histogram of samples
                failed_base = failed_base.T                                                                 # Transpose for visual accuracy
                ax_f[row, col].pcolormesh(                                                                  # Plotting the recovery rate
                    *mesh_edge, failed_base,                                                                # Axes and data
                    cmap = CMAP_2D_3, edgecolor = "face"                                                    # Color and filling mode between pixels
                )
            
            elif row < col:
                param_row, param_col = rows[row, col], cols[row, col]                                       # Parameter names
                edges = (self.edges[param_row], self.edges[param_col])                                      # Edges of the bins
                mesh_edge = np.meshgrid(*edges)

                sample_row = self.samples[param_row][self.is_planet].numpy()                                # Sample row
                sample_col = self.samples[param_col][self.is_planet].numpy()                                # Sample col

                baseline, _, _ = np.histogram2d(sample_row, sample_col, bins = edges)                       # Histogram of samples
                baseline = baseline.T

                mesh_baseline = ax[row, col].pcolormesh(                                                    # Plotting the recovery rate
                    *mesh_edge, baseline,
                    cmap = CMAP_2D_2, edgecolor = "face"
                )
                ax_f[row, col].pcolormesh(                                                                  # Plotting the failed distribution
                    *mesh_edge, baseline,
                    cmap = CMAP_2D_2, edgecolor = "face"
                )
            
        #ax[0, 0].tick_params(axis = "y", which = "both", left = False, labelleft = False)
        fig.colorbar(mesh, label = "Recovered", ax = ax[:, -1], aspect = 50, format = PercentFormatter(decimals = 0))

        self.save_fig(fig, "recovery_2d")
        self.save_fig(fig_f, "failed_2d")
        pl.close()
    
    def plot_stellar_recovery_2d(self) -> None:

        print(PLOTTING, "Stellar recovery 2D")

        ls = np.array([*self.edges_stellar.keys()]) # Getting the names of the parameters to mesh
        rows, cols = np.meshgrid(ls, ls)            # Creating the rows and cols names for corner

        # Recovered fraction and original distribution
        fig, ax = pl.subplots(
            nrows = rows.shape[0], ncols = cols.shape[1],
            sharex = "col", sharey = "row",
            layout = "constrained", figsize = (6.4 * 3, 5.5 * 3)
        )

        # Going through each of the subplots
        for row, col in np.ndindex(rows.shape):
            
            # Diagonal -> 1d distribution
            if row == col:
                #ax[row, col].get_shared_y_axes().remove(ax[row, col])        # Decoupling the ax from the shared y axis
                ax[-1, col].set_xlabel(self.x_labels_stellar[rows[row, col]])
                ax[row, 0].set_ylabel(self.x_labels_stellar[cols[row, col]])

                param = rows[row, col]
                sample = self.samples_stellar[param]

                # Possibility to pass the count on the right ax
                right_ax = ax[row, col].twinx()

                actual, _, _ = right_ax.hist(x = sample[self.is_binary], bins = self.edges_stellar[param], color = CMAP_2D(0.25), label = "Actual")
                _, _, _ = right_ax.hist(
                    x = sample[self.is_recovered_binary],
                    bins = self.edges_stellar[param],
                    color = CMAP_2D(0.9),
                    label = "Recovered"
                )

                right_ax.tick_params(axis = "y", which = "both", right = False, labelright = False)

                if row == 0:
                    right_ax.legend(loc = "upper right")
            
            # Lower left corner -> 2d fraction recovered
            elif row > col:
                param_row, param_col = rows[row, col], cols[row, col]                                       # Parameter names
                edges = (self.edges_stellar[param_row], self.edges_stellar[param_col])                      # Edges of the bins
                mesh_edge = np.meshgrid(*edges)

                sample_row = self.samples_stellar[param_row][self.is_binary].numpy()                        # Sample row
                sample_col = self.samples_stellar[param_col][self.is_binary].numpy()                        # Sample col

                recovered_row = self.samples_stellar[param_row][self.is_recovered_binary].numpy()           # Recovered row
                recovered_col = self.samples_stellar[param_col][self.is_recovered_binary].numpy()           # Recovered col

                baseline, _, _ = np.histogram2d(sample_row, sample_col, bins = edges)                       # Histogram of samples
                recovered, _, _ = np.histogram2d(recovered_row, recovered_col, bins = edges)                # Histogram of recovery
                recovered_frac = (100 * np.divide(recovered, baseline, where = baseline != 0)).round(1)     # Fraction recovered / samples
                recovered_frac = ma.masked_where(baseline == 0, recovered_frac)                             # Masking where no data
                recovered_frac = recovered_frac.T                                                           # Transpose for visual accuracy

                mesh = ax[row, col].pcolormesh(                                                             # Plotting the recovery rate
                    *mesh_edge, recovered_frac,                                                             # Axes and data
                    cmap = CMAP_2D, edgecolor = "face",                                                     # Color and filling mode between pixels
                    vmin = 0, vmax = 100                                                                    # Ensuring min and max between 0% and 100%
                )
            
            elif row < col:
                param_row, param_col = rows[row, col], cols[row, col]                                       # Parameter names
                edges = (self.edges_stellar[param_row], self.edges_stellar[param_col])                      # Edges of the bins
                mesh_edge = np.meshgrid(*edges)

                sample_row = self.samples_stellar[param_row][self.is_binary].numpy()                        # Sample row
                sample_col = self.samples_stellar[param_col][self.is_binary].numpy()                        # Sample col

                baseline, _, _ = np.histogram2d(sample_row, sample_col, bins = edges)                       # Histogram of samples
                baseline = baseline.T

                mesh_baseline = ax[row, col].pcolormesh(                                                    # Plotting the recovery rate
                    *mesh_edge, baseline,
                    cmap = CMAP_2D_2, edgecolor = "face"
                )
            
        fig.colorbar(mesh, label = "Recovered [%]", ax = ax[:, -1], aspect = 50)

        self.save_fig(fig, "recovery_2d_binaries")
        pl.close()

    def plot_recovery(self) -> None:

        thr_keys = [*self.fap_thr.keys()]

        print(PLOTTING, "Validation sample")

        ###############
        # Plots the sample distribution in the validation set

        bins_edges = self.edges["transit_snr"]  # Creating the bins edges to use

        """ with open(os.path.join(self.path, "sample.pt"), "wb+") as f:
            torch.save(sample, f) """

        # Plotting the sample distribution
        fig, ax = pl.subplot_mosaic([["sample"]], layout = "constrained")

        sample_hist, _, _ = ax["sample"].hist(
            x = self.samples["transit_snr"][self.is_planet],
            bins = bins_edges, **SAMPLE_KWARGS
        )

        ax["sample"].set(xlabel = "Transit SNR", ylabel = "# of occurrences", xlim = (0, 201))

        ax["sample"].arrow(160, 30, 35, 0, color = "k", width = 0.4, length_includes_head = True, overhang = 0.2)
        ax["sample"].text(177.5, 31, "Axis continues", ha = "center", va = "baseline")

        self.save_fig(fig, "distribution_sample_snr")
        pl.close()

        ###############
        # Plots the variations between thresholds for the last epoch

        print(PLOTTING, "Last epoch threshold variations")

        sample_single = self.samples["transit_snr"][self.is_planet]

        rec_hist = []
        for thr in cst.THRESHOLDS:
            thr = thr.item()

            is_recovered = self.stats[thr]["found"].flatten(start_dim = 1)[-1][self.is_planet]
            res = sample_single[is_recovered].histogram(bins_edges)[0]
            rec_hist.append(res)
        
        rec_hist = torch.stack(rec_hist)
        sample_hist = torch.stack([sample_single.histogram(bins_edges)[0] for _ in cst.THRESHOLDS])
        ratio = ma.masked_invalid(100 * rec_hist / sample_hist)

        # Computing Bins midpoints
        centered_bins = torch.tensor([(bins_edges[i] + bins_edges[i+1]) / 2 for i in range(len(bins_edges) - 1)])

        # Combining the axes to generate the correct ticks in the figure
        thresholds, edges = torch.meshgrid((cst.THRESHOLDS, centered_bins), indexing="ij")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")

        ax.pcolor(
            edges, thresholds, torch.ones_like(edges), edgecolors = "black",
            cmap = ListedColormap(['none']), hatch = "////"
        )
        nax = ax.twinx()
        mesh_r = nax.pcolor(edges, thresholds, ratio, cmap = CMAP_2D, vmin = 0, vmax = 100)
        fig.colorbar(mesh_r, label = "Recovered [%]")

        far_val = [f"{v[-1]:.02f}" for v in self.fap_thr.values()]
        pos_label_far = (1.09, -0.007)

        nax.set_yticks(thr_keys, far_val)
        nax.set_ylabel("FAR [%]", rotation = 0, fontsize = 10)
        nax.yaxis.set_label_coords(*pos_label_far)

        fig.supxlabel("Transit SNR")
        ax.set_ylabel("Cutoff threshold")

        self.save_fig(fig, "distribution_last_SNR")
        pl.close()

        ###############
        # Plots the evolution of the detection rate with the last best threshold

        print(PLOTTING, "Best threshold evolution for each epoch")

        all_rec = {}
        for thr in cst.THRESHOLDS:
            thr = thr.item()

            rec_hist = []
            for e in self.epochs:

                is_recovered = self.stats[thr]["found"].flatten(start_dim = 1)[e - 1][self.is_planet]
                res = sample_single[is_recovered].histogram(bins_edges)[0]
                rec_hist.append(res)

            all_rec[thr] = torch.stack(rec_hist)
        
        rec_hist = all_rec[self.best_thr]
        sample_hist = torch.stack([sample_single.histogram(bins_edges)[0] for _ in self.epochs])
        ratio = ma.masked_invalid(100 * rec_hist / sample_hist)

        # Combining the axes to generate the correct ticks in the figure
        epochs, edges = torch.meshgrid((self.epochs.to(torch.float32), centered_bins), indexing = "ij")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")
        ax.yaxis.set_major_locator(MaxNLocator(integer = True))

        ax.pcolor(
            edges, epochs, torch.ones_like(edges), edgecolors = "black",
            cmap = ListedColormap(['none']), hatch = "////"
        )
        mesh_r = ax.pcolormesh(edges, epochs, ratio, cmap = CMAP_2D, vmin = 0, vmax = 100)
        fig.colorbar(mesh_r, label = "Recovered [%]")

        fig.supxlabel("Transit SNR")
        ax.set_ylabel("Epoch")

        self.save_fig(fig, "distribution_epoch_SNR")
        pl.close()

        ###############
        # Plots the sample distribution in the validation set

        bins_edges = self.edges["transit_depth"]  # Creating the bins edges to use

        # Plotting the sample distribution
        fig = pl.figure(layout = "constrained")

        sample_hist, _, _ = pl.hist(
            x = self.samples["transit_depth"][self.is_planet],
            bins = bins_edges, **SAMPLE_KWARGS
        )

        pl.xlabel("Transit depth [ppm]")
        pl.ylabel("# of occurrences")

        self.save_fig(fig, "distribution_sample_depth")
        pl.close()

        ###############
        # Plots the variations between thresholds for the last epoch

        print(PLOTTING, "Last epoch threshold variations")

        sample_single = self.samples["transit_depth"][self.is_planet]

        rec_hist = []
        for thr in cst.THRESHOLDS:
            thr = thr.item()

            is_recovered = self.stats[thr]["found"].flatten(start_dim = 1)[-1][self.is_planet]
            res = sample_single[is_recovered].histogram(bins_edges)[0]
            rec_hist.append(res)
        
        rec_hist = torch.stack(rec_hist)
        sample_hist = torch.stack([sample_single.histogram(bins_edges)[0] for _ in cst.THRESHOLDS])
        ratio = ma.masked_invalid(100 * rec_hist / sample_hist)

        # Computing Bins midpoints
        centered_bins = torch.tensor([(bins_edges[i] + bins_edges[i+1]) / 2 for i in range(len(bins_edges) - 1)])

        # Combining the axes to generate the correct ticks in the figure
        thresholds, edges = torch.meshgrid((cst.THRESHOLDS, centered_bins), indexing="ij")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")

        ax.pcolor(
            edges, thresholds, torch.ones_like(edges), edgecolors = "black",
            cmap = ListedColormap(['none']), hatch = "////"
        )
        nax = ax.twinx()
        mesh_r = nax.pcolor(edges, thresholds, ratio, cmap = CMAP_2D, vmin = 0, vmax = 100)
        fig.colorbar(mesh_r, label = "Recovered", format = PercentFormatter())
        ax.set(
            xlabel = "Transit depth [ppm]", ylabel = "Cutoff threshold",
            xticks = [100, 200, 300, 400, 500, 600, 700, 800]
        )

        far_val = [f"{v[-1]:.02f}%" for v in self.fap_thr.values()]
        pos_label_far = (1.09, -0.007)

        nax.set_yticks(thr_keys, far_val)
        nax.set_ylabel("FAR", rotation = 0, fontsize = 10)
        nax.yaxis.set_label_coords(*pos_label_far)

        self.save_fig(fig, "distribution_last_depth")
        pl.close()

        ###############
        # Plots the sample distribution in the validation set

        bins_edges = self.edges["radius_planet"]  # Creating the bins edges to use

        # Plotting the sample distribution
        fig = pl.figure(layout = "constrained")

        sample_hist, _, _ = pl.hist(
            x = self.samples["radius_planet"][self.is_planet],
            bins = bins_edges, **SAMPLE_KWARGS
        )

        pl.xlabel("Planet radii [$R_\oplus$]")
        pl.ylabel("# of occurrences")

        self.save_fig(fig, "distribution_sample_radius")
        pl.close()

        ###############
        # Plots the variations between thresholds for the last epoch

        print(PLOTTING, "Last epoch threshold variations")

        sample_single = self.samples["radius_planet"][self.is_planet]

        rec_hist = []
        for thr in cst.THRESHOLDS:
            thr = thr.item()

            is_recovered = self.stats[thr]["found"].flatten(start_dim = 1)[-1][self.is_planet]
            res = sample_single[is_recovered].histogram(bins_edges)[0]
            rec_hist.append(res)
        
        rec_hist = torch.stack(rec_hist)
        sample_hist = torch.stack([sample_single.histogram(bins_edges)[0] for _ in cst.THRESHOLDS])
        ratio = ma.masked_invalid(100 * rec_hist / sample_hist)

        # Computing Bins midpoints
        centered_bins = torch.tensor([(bins_edges[i] + bins_edges[i+1]) / 2 for i in range(len(bins_edges) - 1)])

        # Combining the axes to generate the correct ticks in the figure
        thresholds, edges = torch.meshgrid((cst.THRESHOLDS, centered_bins), indexing="ij")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")

        ax.pcolor(
            edges, thresholds, torch.ones_like(edges), edgecolors = "black",
            cmap = ListedColormap(['none']), hatch = "////"
        )
        nax = ax.twinx()
        mesh_r = nax.pcolor(edges, thresholds, ratio, cmap = CMAP_2D, vmin = 0, vmax = 100)
        fig.colorbar(mesh_r, label = "Recovered [%]")

        nax.set_yticks(thr_keys, far_val)
        nax.set_ylabel("FAR [%]", rotation = 0, fontsize = 10)
        nax.yaxis.set_label_coords(*pos_label_far)

        fig.supxlabel("Planet radii [$R_\oplus$]")
        ax.set_ylabel("Cutoff threshold")

        self.save_fig(fig, "distribution_last_radius")
        pl.close()

        ###############
        # Plots the evolution of the detection rate with the last best threshold

        print(PLOTTING, "Best threshold evolution for each epoch")

        all_rec = {}
        for thr in cst.THRESHOLDS:
            thr = thr.item()

            rec_hist = []
            for e in self.epochs:

                is_recovered = self.stats[thr]["found"].flatten(start_dim = 1)[e - 1][self.is_planet]
                res = sample_single[is_recovered].histogram(bins_edges)[0]
                rec_hist.append(res)

            all_rec[thr] = torch.stack(rec_hist)
        
        rec_hist = all_rec[self.best_thr]
        sample_hist = torch.stack([sample_single.histogram(bins_edges)[0] for _ in self.epochs])
        ratio = ma.masked_invalid(100 * rec_hist / sample_hist)

        # Combining the axes to generate the correct ticks in the figure
        epochs, edges = torch.meshgrid((self.epochs.to(torch.float32), centered_bins), indexing = "ij")

        fig, ax = pl.subplots(nrows = 1, ncols = 1, layout = "constrained")
        ax.yaxis.set_major_locator(MaxNLocator(integer = True))

        ax.pcolor(
            edges, epochs, torch.ones_like(edges), edgecolors = "black",
            cmap = ListedColormap(['none']), hatch = "////"
        )
        mesh_r = ax.pcolormesh(edges, epochs, ratio, cmap = CMAP_2D, vmin = 0, vmax = 100)
        fig.colorbar(mesh_r, label = "Recovered [%]")

        fig.supxlabel("Planet radii [$R_\oplus$]")
        ax.set_ylabel("Epoch")

        self.save_fig(fig, "distribution_epoch_radius")
        pl.close()

        ###############
        # Plots the ROC

        acc_rec, acc_fap = [], []
        cor_rec = []
        for thr in all_rec.keys():  # Ensuring the correct order
            acc_rec.append( all_rec[thr].sum(1) )                   # Summing the number of recovered planets
            cor_rec.append( all_rec[thr] )                          # Getting the recovery per bin
            acc_fap.append( torch.from_numpy(self.fap_thr[thr]) )   # Getting the FAR

        rec_tot = torch.stack( acc_rec )
        fap_tot = torch.stack( acc_fap )
        sam_tot = sample_hist.sum(1)                                # Summing the number of events in the sample
        ratio_tot = ma.masked_invalid(100 * rec_tot / sam_tot)      # Ratio of recovered to events in sample
        cor_ratio = torch.stack( cor_rec ) / sample_hist            # Ratio per bin
        cor_ratio = 100 * cor_ratio.nanmean(dim = -1)               # Computing corrected recovery

        fig, ax = pl.subplot_mosaic([["ROC"]], layout = "constrained")

        ax["ROC"].plot(fap_tot, ratio_tot, c = "crimson", alpha = 0.2, lw = 0.75, label = "Other")
        ax["ROC"].plot(fap_tot[:,-1], ratio_tot[:,-1], c = "crimson", label = "Best", lw = 1.25)

        ax["ROC"].plot(fap_tot, cor_ratio, c = "purple", alpha = 0.2, lw = 0.75, label = "Other")
        ax["ROC"].plot(fap_tot[:,-1], cor_ratio[:,-1], c = "purple", label = "Best", lw = 1.25)

        xlim = (0, fap_tot[0,-1] * 1.05)
        _min_comb, _max_comb = min(ratio_tot[-1,-1], cor_ratio[-1,-1]), max(ratio_tot[0,-1], cor_ratio[0,-1])
        _yd = (_max_comb - _min_comb) * 0.05
        ylim = (_min_comb - _yd, _max_comb + _yd)

        ax["ROC"].set(xlabel = "FAR", ylabel = "Recovered", xlim = xlim, ylim = ylim)
        ax["ROC"].xaxis.set_major_formatter(PercentFormatter())
        ax["ROC"].yaxis.set_major_formatter(PercentFormatter())
        #ax["ROC"].legend(title = "Epochs", loc = "upper left", markerfirst = False, edgecolor = "k")

        self.save_fig(fig, "roc")
        pl.close()

        for (thr, recovery, far) in zip(cst.THRESHOLDS, ratio_tot, fap_tot):
            self.df_rec_far[f"Recovery {thr:.2}"] = recovery
            self.df_rec_far[f"FAR {thr:.2}"] = far

########################################################################################################################
# The callable function

def analyze(args: str | tuple[str], cfg: dict) -> None:
    
    out_name = os.path.join(cfg["path_output"], "2*")
    outputs = sorted(glob.iglob(out_name))
    target = []

    # We check the possible values
    match args[0]:
        case "last":                        # If last, we only analyze the last one
            target.append( outputs[-1] )
        
        case "all":                         # If all we analyze all the results
            target = outputs
        
        case _:                             # If not we get the requested ones
            target = args

    for p in target:
        result = Output(path = p)
        result.process()
        
########################################################################################################################
