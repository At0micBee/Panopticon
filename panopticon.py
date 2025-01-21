"""
# Panopticon
"""

########################################################################################################################

# Custom modules
from src import model               # ML model
from src import utils               # Utilities
from src.model import ignite_run    # Pytorch Ignite
from src import constants as cst    # Constant values

# Standard python modules
import os                           # System operations
import toml                         # TOML file parser
import shutil                       # System utilities
import logging                      # Log module for python
import argparse                     # Command line argument parser
from os import path                 # Path manipulation
from datetime import datetime       # Date and time manipulation

########################################################################################################################

def create_env(args, cfg: dict, ts: datetime) -> tuple[logging.Logger, str]:
    """
    # Environment creation

    When the program is called in a mode that will create outputs, we need to create the associated environment.

    ## Inputs
    - `cfg`: The configuration file loaded
    - `ts`: The stamp of the program start

    Returns the logger for the run and the output directory path.
    """

    out_dir = path.join(cfg["path_output"], f"{ts}")                    # Naming the run
    os.mkdir(out_dir)                                                   # We create the output dir
    shutil.copy(args.cfg, path.join(out_dir, "config.toml"))            # We make a copy of the configuration used

    # Creating the logger file properly
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    handler = logging.FileHandler(path.join(out_dir, "progress.log"))   # Creating a handler for the I/O
    handler.setLevel(logging.INFO)                                      # INFO output level
    handler.setFormatter(formatter)                                     # Applying formatter

    logger = logging.getLogger("Panopticon")                            # Naming the logger
    logger.setLevel(logging.INFO)                                       # INFO output level
    logger.addHandler(handler)                                          # Applying the handler

    return logger, out_dir

########################################################################################################################

def main() -> None:

    time_start = datetime.now()                             # Marking the start of the program

    ##################################################
    # Defining and parsing the commands
    ##################################################

    parser = argparse.ArgumentParser()                      # Creating the command line parser

    ##########
    # Program modes arguments

    parser.add_argument(                                    # Model info
        "--info",
        type = int,
        default = None,
        nargs = "+",
        help = "Creates a summary of the model parameters."
    )

    parser.add_argument(                                    # Plotting from ftr
        "--plot",
        type = int,
        default = None,
        nargs = "+",
        help = "Calls the program in plotting mode."
    )

    parser.add_argument(                                    # Plotting all from ftr
        "--plotall",
        action = "store_true",
        help = "Calls the program in plotting mode."
    )

    parser.add_argument(                                    # Label maker
        "--labels",
        action = "store_true",
        help = "Generates the labels, based on the AllParameters files."
    )

    parser.add_argument(                                    # Prepare the data for training
        "--prepare",
        action = "store_true",
        help = "Prepares the data to the final format for training. Requires labels to have been made."
    )

    parser.add_argument(                                    # Plots packaged light curves
        "--inspect",
        type = int,
        default = None,
        nargs = "+",
        help = "Plots the final packaged data."
    )

    parser.add_argument(                                    # Checks for invalid boxes in light curves
        "--invalid_detector",
        action = "store_true",
        help = "Runs the invalid detection tool on the prepared set."
    )

    parser.add_argument(                                    # Removes the invalid light curves
        "--invalid_cleaner",
        action = "store_true",
        help = "Removes the erroneous files from the dataset (based on --invalid_detector)."
    )

    parser.add_argument(                                    # Model trainer
        "--train",
        action = "store_true",
        help = "Starts the training."
    )

    parser.add_argument(                                    # Data analyzer
        "--analyze",
        type = str,
        default = None,
        nargs = "+",
        help = "Analyzes the results of an output."
    )

    ##########
    # Optional arguments

    parser.add_argument(                                    # Config file
        "--cfg",
        type = str,
        default = cst.PATH_CONFIG,
        help = "Specify a config file (full path). If nothing, uses the default config."
    )

    parser.add_argument(                                    # Do we augment the prepared data
        "--augment",
        action = "store_true",
        help = "Whether to augment the data by flipping time-wise (with --prepare)."
    )

    parser.add_argument(                                    # Do we filter the inf in the data
        "--filter_inf",
        action = "store_true",
        help = "Whether to remove all light curves with inf values (with --prepare)."
    )

    args = parser.parse_args()                              # Parsing the arguments

    new_run = (args.info is not None) | args.train          # Checking the requirements for a new run

    ##################################################
    # We handle the basic files
    ##################################################

    with open(args.cfg, "r") as f:                                      # We load the configuration file
        cfg: dict = toml.load(f)                                        # Parsing yaml file

    # We change the output path if necessary
    if cfg["path_output"] == "":                                        # If nothing specified
        cfg["path_output"] = cst.PATH_OUT                               # We use the constant value (./out/)

    # We check if the requested mode needs the creation of the environment
    if new_run:                                                         # Mode requiring creation: Info, Train
        logger, out_dir = create_env(args, cfg = cfg, ts = time_start)  # We create the env
        logger.info("Output environment created")

    ##################################################
    # We run the program depending on the input
    ##################################################

    # Printing model info
    if args.info is not None:
        logger.info("Building information of the model")
        sim = model.Run(cfg = cfg, time_start = time_start)
        sim.info(shape = tuple(args.info))
        shutil.rmtree(out_dir)
    
    # Plotting requested from ftr
    if args.plot:
        utils.plot(cfg, args.plot)
    
    # plotting all from ftr
    if args.plotall:
        utils.plotall(cfg)
    
    # Creating the labels
    if args.labels:
        utils.make_labels(cfg)

    # Preparing the training data
    if args.prepare:
        utils.package(cfg, args.augment, args.filter_inf)

    # Inspecting a packaged lightcurve
    if args.inspect:
        utils.inspect(args.inspect, cfg)

    # Invalid detection
    if args.invalid_detector:
        utils.find_invalid(cfg)

    # Invalid removal
    if args.invalid_cleaner:
        utils.clean_invalid(cfg)

    # Starting training
    if args.train:
        logger.info("Program called in training mode")
        sim = model.Run(cfg = cfg, time_start = time_start)
        sim.train()
        #ignite_run.start(cfg = cfg, time_start = time_start)

    # Analyzing the requested run
    if args.analyze is not None:
        utils.analyze(args.analyze, cfg)

    ##################################################
    # Finishing up
    ##################################################

    time_end = datetime.now()                                               # Marking the start of the program
    runtime = time_end - time_start                                         # Computing the runtime
    if new_run:
        logger.info(f"Program finished, runtime {runtime}")                 # Logging final runtime

########################################################################################################################
# Program entry point

if __name__ == "__main__":
    main()

########################################################################################################################
