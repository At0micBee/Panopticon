"""
# Training routines for the models
"""

########################################################################################################################

# Model related modules
from .loss import Loss                                      # The loss function
from .model import load_from_params                         # The ML model itself
from .dataset import LightCurveSet                          # The dataset for the PLATO light curves
from .selector import optimizer, scheduler                  # Selecting the type of optimizer and scheduler used
from ..utils.recovery import transit_recovery               # The transit recovery function
from .. import constants as cst                             # Constant values

# Standard Python modules
import os                                                   # System operations
import toml                                                 # TOML file parser
import logging                                              # Log module for python
from datetime import datetime                               # Date and time manipulation

# Torch and its required subparts
import torch                                                # Machine learning and tensors
from torch.utils.data import DataLoader, random_split       # Dataloader class and random split function
from torchsummary import summary                            # Information about the model used

# Dedicated metrics
from torchmetrics.functional.classification import (        # Importing the metrics (functions)
    binary_accuracy,                                        # Accuracy function
    binary_precision,                                       # Precision function
    binary_recall,                                          # Recall function
    binary_f1_score,                                        # F1 score
    binary_average_precision,                               # AP function
    binary_confusion_matrix,                                # Confusion matrix
    binary_jaccard_index                                    # IOU function
)

########################################################################################################################

class Run:
    """
    # Setting up the running environment for training

    ## Description

    ## Inputs
    - `cfg`: The configuration file loaded
    - `time_start`: The stamp of the program start

    Returns the parameters required to run a training or testing job.
    """

    def __init__(self, cfg: dict, time_start: datetime) -> None:

        ##################################################
        # Initializing base info
        ##################################################

        self.logger = logging.getLogger("Panopticon")                           # Fetching the logger for the program
        self.logger.info("Entering setup!")

        self.cfg = cfg                                                          # Saving the config
        self.time_start = time_start                                            # Saving the start time
        self.path_out = os.path.join(cfg["path_output"], str(self.time_start))  # Creating a unique name for the run
        self.path_models = os.path.join(self.path_out, "model")                 # Creating the path to the models save
        self.path_stats = os.path.join(self.path_out, "stats")                  # Creating the path to the stats save
        
        os.mkdir(self.path_models)                                              # Creating the models output dir
        os.mkdir(self.path_stats)                                               # Creating the stats output dir
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # We check the device availability
        self.logger.info(f"Found device: {self.device}")

        ##################################################
        # Preparing the data sets
        ##################################################

        self.data = LightCurveSet(cfg["path_prepared"])                         # We pass the input files to the Dataset
        self.logger.info(f"[Data] Created complete Dataset ({len(self.data)})") # Logging the total length

        fracs = cfg["fraction"]                                                 # Getting the fractions from the config
        self.logger.info(f"[Data] Requested fractions: {fracs}")                # Logging the values
        self.train_set, self.valid_set = random_split(                          # We split into subsets
            dataset = self.data,                                                # The sets to break down
            lengths = [fracs["train"], fracs["valid"]],                         # The fraction given to each Dataset
            generator = torch.Generator().manual_seed(cfg["seed"])              # Manual seed, for reproducibility
        )

        # If requested we export the dataset files
        if cfg["export_sets"]:

            # We create a dict and save the names
            sets = {
                "training": [ os.path.basename(n) for n, _, _, _ in self.train_set ],
                "validation": [ os.path.basename(n) for n, _, _, _ in self.valid_set ]
            }

            # Writing the dict to a pickle file
            sets_name = os.path.join(self.path_out, "sets.pt")
            with open(sets_name, "wb") as writer:
                torch.save(sets, writer)

        self.train_len, self.valid_len = len(self.train_set), len(self.valid_set)
        self.logger.info(f"[Data] Created training ({self.train_len}) and validation ({self.valid_len}) Subsets")

        ##################################################
        # Preparing the model
        ##################################################
         
        self.net = load_from_params(internals = cfg["model"]).to(self.device)   # Creating the model
        self.logger.info(f"[Model] Model initialized: {cfg['model']['name']}")

        with open(os.path.join(self.path_out, "model_internals.toml"), "w+") as mi:
            toml.dump(self.net.internals, mi)
        self.logger.info("[Model] Exported internal values")

        if cfg["base_model"]:                                                   # Checking if we load an existing model
            with open(cfg["base_model"], "rb") as m:                            # Opening file
                self.net.load_state_dict(torch.load(m))                         # Assigning weights
            self.logger.info(f"[Model] Weights loaded from {cfg['base_model']}")
        
        else:
            self.logger.info("[Model] No weights specified")
        
        ##################################################
        # Preparing the data loaders and loss function
        ##################################################

        self.train_loader = DataLoader(                                         # Training loader
            self.train_set, **self.cfg["dataloader"]["train"]                   # Passing training set and kwargs
        )
        self.logger.info("[Data] Training dataloader created")

        self.valid_loader = DataLoader(                                         # Validation loader
            self.valid_set, **self.cfg["dataloader"]["valid"]                   # Passing validation set and kwargs
        )
        self.logger.info("[Data] Validation dataloader created")

        self.loss_function = Loss(cfg["model"]["loss"], self.device)            # Loss function
        self.logger.info("[Loss] Loss function initialized")
    
    ############################################################
    # Training function

    def train(self) -> None:
        """
        # Training the model

        Using the requested dataset (split into a training and validation `Subsets`), we update the model's weights.
        All the relevant parameters are supplied on the `Run` class, and the `train` function therefore requires
        no additional parameters.
        """

        ##################################################
        # Optim and scheduler
        ##################################################

        self.optimizer = optimizer[self.cfg["optimizer"]["name"]](              # Optimizer
            self.net.parameters(), **self.cfg["optimizer"]["kwargs"]            # Passing model and kwargs
        )
        self.logger.info(f"[Optimizer] Optimizer initialized ({self.cfg['optimizer']['name']})")

        self.scheduler = scheduler[self.cfg["scheduler"]["name"]](              # Scheduler
            self.optimizer, **self.cfg["scheduler"]["kwargs"]                   # Passing optimizer and kwargs
        )
        self.logger.info(f"[Scheduler] Scheduler initialized ({self.cfg['scheduler']['name']})")

        out_name = os.path.join(self.path_models, f"{0:04}.pt")                 # Creating the proper path
        with open(out_name, "wb") as writer:                                    # Opening as binary
            torch.save(self.net.state_dict(), writer)                           # Saving the original model
        
        ##################################################
        # Running the epochs
        ##################################################

        for epoch in range(1, self.cfg["training_epochs"] + 1):

            time_epoch = datetime.now()
            time_since_start = time_epoch - self.time_start
            self.epoch_prog = f"Epoch {epoch:03} / {self.cfg['training_epochs']:03}"
            self.logger.info(f"{self.epoch_prog} :: starting, {time_since_start} elapsed since start")

            stats = {}                                                          # Creating the dict to keep track of data

            running_loss = 0                                                    # Initial total loss
            self.net.train()                                                    # Activating training for model

            # We iterate through the training set
            for f_name, _, lc, ref in self.train_loader:
                
                self.optimizer.zero_grad(set_to_none = True)                    # Zeroing the optimizer

                # Sending LC and ref to the correct type
                lc = lc.to(device = self.device, non_blocking = True)
                ref = ref.to(device = self.device, non_blocking = True)
                ref_int = ref.to(dtype = int, non_blocking = True)

                prediction = self.net(lc)                                       # Computing the prediction from the model

                loss = self.loss_function(prediction, ref)                      # Estimating the loss compared to reality

                loss.backward()                                                 # Propagating the loss
                self.optimizer.step()                                           # Incrementing the model

                # Computing training accuracy and saving to dict
                with torch.no_grad():
                    accuracy = binary_accuracy(prediction, ref_int, 0.5, "samplewise").detach().cpu()
                    stats[tuple(f_name)] = { "accuracy": accuracy }

                running_loss += loss.item()                                     # Computing full loss of epoch
            
            # Saving model training stats and model params

            out_name = os.path.join(self.path_stats, f"train_{epoch:04}.pt")    # Creating the proper path
            with open(out_name, "wb") as writer:                                # Opening as Binary
                torch.save(stats, writer)                                       # Saving the parameters
            
            out_name = os.path.join(self.path_models, f"{epoch:04}.pt")         # Creating the proper path
            with open(out_name, "wb") as writer:                                # Opening as binary
                torch.save(self.net.state_dict(), writer)                       # Saving the original model
            self.logger.info(f"{self.epoch_prog} :: Exported model parameters")
            
            self.logger.info(f"{self.epoch_prog} :: finished training, entering evaluation (average loss: {running_loss})")

            # Validation

            self.validate(epoch)                                                # We call the validation routine

            # Updating scheduler if any
            if self.cfg["scheduler"]["use"]:                                    # Checking if we need the scheduler

                if self.cfg["scheduler"]["name"] == "ReduceLROnPlateau":
                    self.scheduler.step(loss)                                   # Incrementing the scheduler
                else:
                    self.scheduler.step()
                
                current_lr = self.scheduler.get_lr()
                self.logger.info(f"[Scheduler] {self.epoch_prog} :: Learning rate adjusted -> {current_lr}")
            
            self.logger.info(f"{self.epoch_prog} :: End of epoch")

    ############################################################
    # Validation function

    @torch.no_grad()                                                            # Disabling gradient computation
    def validate(self, epoch: int) -> None:
        """
        # Testing a model on a given `Subset`

        We go through the validation dataset to test the performance of the model.
        In standard operation, the function is called at every training epoch.

        The function uses the `@torch.no_grad()` decorator to disable `Tensor` gradient computation,
        which isn't necessary during testing. The model is also flipped in `model.eval()` mode at the
        beginning of the function, and set back to `model.train()` mode before returning.

        - `epoch`: the current epoch counter, for book-keeping.
        """

        self.net.eval()                                                         # Setting mode to evaluation mode
        self.logger.info(f"{self.epoch_prog} :: Entered validation step")

        stats = {}                                                              # Creating the dict to keep track of data
        running_loss = 0                                                        # Initial total loss

        for f_name, sims_params, lc, ref in self.valid_loader:                  # Iterating through the loader
            print(f_name)

            stats[tuple(f_name)] = {}                                           # Creating the dict at this value

            lc = lc.to(device = self.device, non_blocking = True)               # Sending lc tensor to the right device
            ref = ref.to(device = self.device, non_blocking = True)             # Same for the reference
            ref_int = ref.to(dtype = int, non_blocking = True)                  # Creating the reference version with int

            # Prediction and associated loss
            prediction = self.net(lc)                                           # Model prediction
            loss = self.loss_function(prediction, ref).detach().cpu()           # Estimating the loss
            running_loss += loss.item()                                         # Computing full loss of epoch

            # We compute the metrics for each threshold
            for thr in cst.THRESHOLDS:

                thr = thr.item()                                                # Extracting the float in the tensor

                # Computing metrics
                accuracy = binary_accuracy(prediction, ref_int, thr, "samplewise").detach().cpu()       # Computing accuracy
                precision = binary_precision(prediction, ref_int, thr, "samplewise").detach().cpu()     # Computing precision
                recall = binary_recall(prediction, ref_int, thr, "samplewise").detach().cpu()           # Computing the recall
                f1 = binary_f1_score(prediction, ref_int, thr, "samplewise").detach().cpu()             # Computing the F1
                ap = binary_average_precision(prediction, ref_int, [thr]).detach().cpu()                # Computing the average precision
                cm = binary_confusion_matrix(prediction, ref_int, thr).detach().cpu()                   # Calculating the confusion matrix
                iou = binary_jaccard_index(prediction, ref_int, thr).detach().cpu()                     # Computing the IOU
                sample, found, iou_cm = transit_recovery(prediction, ref, sims_params, thr)             # List of the retrieved radii

                stats[tuple(f_name)][thr] = {                                   # We create a dict with the params
                    "loss": loss,                                               # Adding the loss to the dict
                    "accuracy": accuracy,                                       # Accuracy
                    "precision": precision,                                     # Precision
                    "recall": recall,                                           # Recall
                    "f1": f1,                                                   # F1
                    "ap": ap,                                                   # Average precision
                    "cm": cm,                                                   # Confusion matrix
                    "iou": iou,                                                 # IOU
                    "found": found,                                             # The map of the found planets
                    "iou_cm": iou_cm                                            # Reduced confusion matrix
                }
                stats[tuple(f_name)][thr].update(sample)                        # Adding the samples to the output

        self.logger.info(f"{self.epoch_prog} :: Validation finished, average loss: {running_loss / len(self.valid_loader)}")

        out_name = os.path.join(self.path_stats, f"valid_{epoch:04}.pt")        # Creating the proper path
        with open(out_name, "wb") as writer:                                    # Opening as Binary
            torch.save(stats, writer)                                           # Saving the parameters

        self.net.train()                                                        # Returning the model to training
    
    ############################################################

    def info(self, shape: list[int]) -> None:
        """
        # Prints out info about the model

        We show the default version, using `print(model)`, and a more detailed implementation in the
        `torchsummary` module.
        """
        print(self.net)
        summary(self.net, shape, device = str(self.device))

########################################################################################################################
