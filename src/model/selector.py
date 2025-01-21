"""
# Listing options for optimizer and scheduler
"""

########################################################################################################################

# Torch and its required subparts
import torch                        # Machine learning

########################################################################################################################

optimizer = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
    "SGD": torch.optim.SGD,
    "SparseAdam": torch.optim.SparseAdam
}

scheduler = {
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "StepLR": torch.optim.lr_scheduler.StepLR
}

########################################################################################################################
