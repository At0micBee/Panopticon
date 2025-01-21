#
# Computes the recovery (transit-wise) of a model
#

########################################################################################################################

# Model related modules
from .. import constants as cst                         # Constant values

# Torch and its required subparts
import torch                                            # Machine learning and tensors
from torch import Tensor                                # The tensor type
from torchvision.ops import box_iou                     # The box iou function

########################################################################################################################
# Required functions

@torch.no_grad()
def get_boxes(arr: Tensor, thr: float) -> Tensor:
    """
    # Computes the boxes around a transit event

    Note that we return a 2D box, with fictitious coordinates in `y`. This allows to simply call the
    `torch` implementation of the IOU computation for bounding boxes: `box_iou`. We always return 
    `ymin, ymax = 0, 1` so as to always have a perfect match in that dimension.

    - `arr`: the tensor with the detection probability, shape [N]
    - `thr`: the threshold for detection [0, 1]
    
    Returns a tensor of shape [N, 4], where each block is the coordinates of the bounding box.
    """

    test = torch.where(arr > thr)[0]                            # Checking if there are point above the threshold level
    boxes = []                                                  # Empty TCE list

    if len(test) == 0:                                          # If no prediction of transit
        return None                                             # We return None

    skips = test.diff()                                         # Checks to see delta between two marked points
    jumps = torch.where(skips > cst.STD_DURATION)[0]            # If there is a jump by more that the lim, it's another event

    start = 0                                                   # We start the count at zero
    for j in jumps:                                             # If there are jumps, there are more than 1 event
        boxes.append(                                           # Appending all the boxes
            (test[start].item(), 0, test[j].item(), 1)          # We use torch.box_iou -> needs 4 points
        )
        start = j + 1                                           # Incrementing the start to the next box
        
    boxes.append(
        (test[start].item(), 0, test[-1].item(), 1)             # Adding last box, or the only box if no jumps
    )
    
    return torch.tensor(boxes)                                  # Returning all the boxes

###############

@torch.no_grad()
def transit_recovery(
        prediction: Tensor,
        reference: Tensor,
        sims_params: dict,
        thr: float) -> tuple[dict, Tensor, Tensor]:
    """
    # Evaluates the bounding box of a transit

    Estimates whether the transit was retrieved in the data. Returns the list of radii that were found,
    as well as the list of transit that were present in the dataset.
    """

    # Initializing the values
    sample = { param: [] for param in cst.SIMS_PARAMS } # Empty arrays for sample
    cm, found = [], []                                  # Empty confusion matrix and found param

    for idx, (pred, ref) in enumerate(zip(prediction, reference)):

        # Adding the ground truth to the sample
        for param, val in sample.items():
            val.append(sims_params[param][idx].item())

        pred, ref = pred[0], ref[0]                     # Getting correct dim

        box_pred = get_boxes(pred, thr)                 # computing pred boxes
        box_ref = get_boxes(ref, thr)                   # Getting ref boxes

        if box_pred is not None:
            iou = box_iou(box_pred, box_ref)            # Computing the IOU matrix

            positive = iou != 0                         # Finding the matches

            fn = ~positive.any(dim = 0)                 # False negative
            tp = positive.any(dim = 1)                  # True positive
            fp = ~tp                                    # False positive

            # Counting the types
            num_fn, num_tp, num_fp = fn.sum(), tp.sum(), fp.sum()
        
        else:
            num_fn, num_tp, num_fp = box_ref.shape[0], 0, 0
        
        # Appending batch recovery
        cm.append( [num_fn, num_tp, num_fp] )           # The confusion matrix
        found.append(num_tp > 0)                        # Whether we found the a transit

    # Renaming keys and casting to tensors
    sample = { f"sample_{param}": torch.tensor(val) for param, val in sample.items() }

    return sample, torch.tensor(found), torch.tensor(cm)

########################################################################################################################
