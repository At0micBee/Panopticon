"""
# Loss functions

This module compiles loss functions that can be used to train the model. They are all written 
as a `torch.nn.Module` class, for interoperability.
"""

########################################################################################################################

# Torch and its required subparts
import torch                                                    # Machine learning and tensors
from torch import Tensor                                        # The tensor type

########################################################################################################################
# Creating the loss function

class _DiceLoss(torch.nn.Module):
    """
    # Dice loss implementation

    Similar (and, in fact, usually equivalent) to the IOU. Allows to compute the loss based on a specific metric
    rather than just the Binary Cross Entropy.
    """

    def __init__(self) -> None:
        super(_DiceLoss, self).__init__()

        self.smooth = 1     # Smoothing window

    def forward(self, predictions: Tensor, reference: Tensor) -> Tensor:

        predictions = predictions[:, 0].contiguous().view(-1)                                           # Re-casting
        reference = reference[:, 0].contiguous().view(-1)                                               # Re-casting
        
        intersection = (predictions * reference).sum()                                                  # I = sum(P*R)
        dice = (2 * intersection + self.smooth) / (predictions.sum() + reference.sum() + self.smooth)   # Computing dice

        return 1 - dice

########################################################################################################################

class Loss(torch.nn.Module):
    """
    # Loss function for Panopticon

    Selects, by name, the loss function to use during training.
    """

    def __init__(self, fn_name: str, device: torch.device) -> None:
        super(Loss, self).__init__()

        match fn_name:

            case "BCELoss":
                fn = torch.nn.BCELoss()
            
            case "BCEWithLogitsLoss":
                fn = torch.nn.BCEWithLogitsLoss()
            
            case "DiceLoss":
                fn = _DiceLoss()

            case _:
                raise ValueError(f"{fn_name} isn't a valid loss function option!")
        
        self.fn = fn.to(device = device)
    
    def forward(self, prediction: Tensor, reference: Tensor) -> Tensor:
        return self.fn(prediction, reference)

########################################################################################################################
