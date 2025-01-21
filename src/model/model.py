"""
# Models

All variants of the model are defined here. The core model is based upon a `Unet` architecture, or
its variants (`Unet++`, `Unet3+`, ...).

The goal of the model it to produce a probability map with a one-to-one correspondance with the input
light curve. Each point is assigned a probability to belong to a transit event. A value of zero indicates
that the point belongs to the continuum, while a value of one means that the point is inside of a transit.

As per `torch` computational format, the required input is of shape `[B, F, P]`, with `B` the batch size,
`F` the number of input features and `P` to length of the time series.
"""

########################################################################################################################

from __future__ import annotations  # Better hinting

# Torch and its required subparts
import torch                        # Machine learning and tensors
from torch import Tensor            # The tensor type

########################################################################################################################
# Functions

def load_from_params(internals: dict = None) -> Panopticon_U | Panopticon_UPP | Panopticon_U3P:
    """
    # Loading the correct model using the internals

    This allows being flexible using state_dict of each model, and avoids
    relying on saving/loading the model directly, which can break things
    very easily.

    It can be used to initialize a model, either by name alone or with a specific internal structure. If 
    the model is specified by name alone, the internal structure will be attributed using the current
    default values in the model definitions.

    - `internals`: the `dict` with the internal parameters of the model (obtained using `model.export_values()`)

    Returns the requested model class, with or without specified internal parameters.
    """

    # List of models
    options = {
        "Panopticon_U": Panopticon_U,
        "Panopticon_UPP": Panopticon_UPP,
        "Panopticon_U3P": Panopticon_U3P
    }

    # Returning the model with parameters
    return options[internals["name"]](internals)

def _select_activation(name: str):
    """
    # Returns the activation layer based on its name
    """

    match name:

        case "Sigmoid":
            activation = torch.nn.Sigmoid()
        
        case "Softmax":
            activation = torch.nn.Softmax(dim = 1)      # The softmax is applied class-wise: [batch_size, class_num, points_num]
        
        case None:
            activation = torch.nn.Identity()

        case _:
            raise ValueError(f"{name} isn't a valid activation layer!")
    
    return activation

########################################################################################################################
# Creating the model

class Panopticon_U(torch.nn.Module):
    """
    # Panopticon, Unet architecture

    Using the Unet architecture, identifies features and returns them on a 1:1 output map.
    """

    _NAME = "Panopticon_U"

    def __init__(self, internals: dict) -> None:
        """
        # Panopticon (Unet) initialization

        - `internals`: the `dict` with the internal parameters of the model (obtained using `model.export_values()`)

        If `internals` is not specified, the model will be created using the default values defined in `__init__()`.
        """
        super(Panopticon_U, self).__init__()

        # Saving the internals
        self.internals = internals

        # Basic shape
        self.channels_in = internals["general"]["channels_in"]
        self.channels_out = internals["general"]["channels_out"]
        self.init_features = internals["general"]["init_features"]
        self.drop_rate = internals["general"]["drop_rate"]

        # Defining parameters
        self.param_conv = internals["convolution"]
        self.param_trans = internals["trans_conv"]

        ##########
        # Encoding the data

        self.pool = torch.nn.MaxPool1d(kernel_size = 2, stride = 2)

        self.encoder_1 = Block(self.channels_in, self.init_features, self.drop_rate, self.param_conv)
        self.encoder_2 = Block(self.init_features, self.init_features * 2, self.drop_rate, self.param_conv)
        self.encoder_3 = Block(self.init_features * 2, self.init_features * 4, self.drop_rate, self.param_conv)
        self.encoder_4 = Block(self.init_features * 4, self.init_features * 8, self.drop_rate, self.param_conv)

        ##########
        # Bottleneck, turning around

        self.bottleneck = Block(self.init_features * 8, self.init_features * 16, self.drop_rate, self.param_conv)

        ##########
        # Decoding the data

        self.trans_conv_4 = torch.nn.ConvTranspose1d(self.init_features * 16, self.init_features * 8, **self.param_trans)
        self.decoder_4 = Block((self.init_features * 8) * 2, self.init_features * 8, self.drop_rate, self.param_conv)

        self.trans_conv_3 = torch.nn.ConvTranspose1d(self.init_features * 8, self.init_features * 4, **self.param_trans)
        self.decoder_3 = Block((self.init_features * 4) * 2, self.init_features * 4, self.drop_rate, self.param_conv)

        self.trans_conv_2 = torch.nn.ConvTranspose1d(self.init_features * 4, self.init_features * 2, **self.param_trans)
        self.decoder_2 = Block((self.init_features * 2) * 2, self.init_features * 2, self.drop_rate, self.param_conv)

        self.trans_conv_1 = torch.nn.ConvTranspose1d(self.init_features * 2, self.init_features, **self.param_trans)
        self.decoder_1 = Block(self.init_features * 2, self.init_features, self.drop_rate, self.param_conv)

        ##########
        # Computing output
        self.conv_out = torch.nn.Conv1d(in_channels = self.init_features, out_channels = self.channels_out, kernel_size = 1)

        self.activation = _select_activation(internals["activation"])                                                   # Activation layer

    ########################################

    def forward(self, x: Tensor) -> Tensor:
        """
        # Forward pass at every call

        Note (from `torch` documentation): this method shouldn't be called directly, but prediction should
        use the module call directly, e.g. `Panopticon_U(x)`, as it ensures all registered hooks are ran properly.

        `x`: the input to evaluate, shape `[B, F, P]` (see module docs for details)

        Returns the class probability for each point.
        """

        enc1 = self.encoder_1(x)
        enc2 = self.encoder_2(self.pool(enc1))
        enc3 = self.encoder_3(self.pool(enc2))
        enc4 = self.encoder_4(self.pool(enc3))
        
        bn = self.bottleneck(self.pool(enc4))

        dec4 = self.trans_conv_4(bn)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.decoder_4(dec4)

        dec3 = self.trans_conv_3(dec4)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.decoder_3(dec3)

        dec2 = self.trans_conv_2(dec3)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.decoder_2(dec2)

        dec1 = self.trans_conv_1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.decoder_1(dec1)

        return self.activation(self.conv_out(dec1))

########################################################################################################################

class Panopticon_UPP(torch.nn.Module):
    """
    # Panopticon, Unet++ architecture

    Using the Unet++ architecture, identifies features and returns them on a 1:1 output map.
    """

    _NAME = "Panopticon_UPP"

    def __init__(self, internals: dict) -> None:
        """
        # Panopticon (Unet++) initialization

        - `internals`: the `dict` with the internal parameters of the model (obtained using `model.export_values()`)

        If `internals` is not specified, the model will be created using the default values defined in `__init__()`.
        """
        super(Panopticon_UPP, self).__init__()

        # Saving the internals
        self.internals = internals

        # Basic shape
        self.channels_in = internals["general"]["channels_in"]
        self.channels_out = internals["general"]["channels_out"]
        self.init_features = internals["general"]["init_features"]
        self.drop_rate = internals["general"]["drop_rate"]

        # Defining parameters
        self.param_conv = internals["convolution"]
        self.param_trans = internals["trans_conv"]

        ##########
        # Creating levels

        # Level 0
        self.conv_in_00 = Block(self.channels_in, self.init_features, self.drop_rate, self.param_conv)                                  # Entry to 0,0

        # Level 1
        self.conv_00_10 = Block(self.init_features, self.init_features * 2, self.drop_rate, self.param_conv)                            # 0,0 to 1,0
        self.conv_01 = Block(self.init_features * 2, self.init_features, self.drop_rate, self.param_conv)                               # [0,0 + U(1,0)]
        self.trcv_10 = torch.nn.ConvTranspose1d(self.init_features * 2, self.init_features, **self.param_trans)         # U(1,0)

        # Level 2
        self.conv_10_20 = Block(self.init_features * 2, self.init_features * 4, self.drop_rate, self.param_conv)                        # 1,0 to 2,0
        self.conv_11 = Block(self.init_features * 2 * 2, self.init_features * 2, self.drop_rate, self.param_conv)                       # [1,0 + U(2,0)]
        self.conv_02 = Block(self.init_features * 3, self.init_features, self.drop_rate, self.param_conv)                               # [0,0 + 0,1 + U(1,1)]
        self.trcv_20 = torch.nn.ConvTranspose1d(self.init_features * 4, self.init_features * 2, **self.param_trans)     # U(2,0)
        self.trcv_11 = torch.nn.ConvTranspose1d(self.init_features * 2, self.init_features, **self.param_trans)         # U(1,1)

        # Level 3
        self.conv_20_30 = Block(self.init_features * 4, self.init_features * 8, self.drop_rate, self.param_conv)                        # 2,0 to 3,0
        self.conv_21 = Block(self.init_features * 4 * 2, self.init_features * 4, self.drop_rate, self.param_conv)                       # [2,0 + U(3,0)]
        self.conv_12 = Block(self.init_features * 2 * 3, self.init_features * 2, self.drop_rate, self.param_conv)                       # [1,0 + 1,1 + U(2,1)]
        self.conv_03 = Block(self.init_features * 4, self.init_features, self.drop_rate, self.param_conv)                               # [0,0 + 0,1 + 0,2 + U(1,2)]
        self.trcv_30 = torch.nn.ConvTranspose1d(self.init_features * 8, self.init_features * 4, **self.param_trans)     # U(3,0)
        self.trcv_21 = torch.nn.ConvTranspose1d(self.init_features * 4, self.init_features * 2, **self.param_trans)     # U(2,1)
        self.trcv_12 = torch.nn.ConvTranspose1d(self.init_features * 2, self.init_features, **self.param_trans)         # U(1,2)

        # Level 4
        self.conv_30_40 = Block(self.init_features * 8, self.init_features * 16, self.drop_rate, self.param_conv)                       # 3,0 to 4,0
        self.conv_31 = Block(self.init_features * 8 * 2, self.init_features * 8, self.drop_rate, self.param_conv)                       # [3,0 + U(4,0)]
        self.conv_22 = Block(self.init_features * 4 * 3, self.init_features * 4, self.drop_rate, self.param_conv)                       # [2,0 + 2,1 + U(3,1)]
        self.conv_13 = Block(self.init_features * 2 * 4, self.init_features * 2, self.drop_rate, self.param_conv)                       # [1,0 + 1,1 + 1,2 + U(2,2)]
        self.conv_04 = Block(self.init_features * 5, self.init_features, self.drop_rate, self.param_conv)                               # [0,0 + 0,1 + 0,2 + 0,3 + U(1,3)]
        self.trcv_40 = torch.nn.ConvTranspose1d(self.init_features * 16, self.init_features * 8, **self.param_trans)    # U(4,0)
        self.trcv_31 = torch.nn.ConvTranspose1d(self.init_features * 8, self.init_features * 4, **self.param_trans)     # U(3,1)
        self.trcv_22 = torch.nn.ConvTranspose1d(self.init_features * 4, self.init_features * 2, **self.param_trans)     # U(2,2)
        self.trcv_13 = torch.nn.ConvTranspose1d(self.init_features * 2, self.init_features, **self.param_trans)         # U(1,3)

        # Convolution for output
        self.conv_out = torch.nn.Conv1d(self.init_features, self.channels_out, 1)                                       # Last,0 to output

        # Max pooling layer, static values so defined only once
        self.pool = torch.nn.MaxPool1d(kernel_size = 2, stride = 2)                                                     # Pooling

        self.activation = _select_activation(internals["activation"])                                                   # Activation layer
    
    ########################################
    
    def forward(self, x: Tensor) -> Tensor:
        """
        # Forward pass at every call

        Note (from `torch` documentation): this method shouldn't be called directly, but prediction should
        use the module call directly, e.g. `Panopticon_UPP(x)`, as it ensures all registered hooks are ran properly.

        `x`: the input to evaluate, shape `[B, F, P]` (see module docs for details)

        Returns the class probability for each point.
        """

        x_00 = self.conv_in_00(x)
        x_10 = self.conv_00_10( self.pool(x_00) )
        x_01 = self.conv_01( torch.cat((x_00, self.trcv_10(x_10)), dim = 1) )

        x_20 = self.conv_10_20( self.pool(x_10) )
        x_11 = self.conv_11( torch.cat((x_10, self.trcv_20(x_20)), dim = 1) )
        x_02 = self.conv_02( torch.cat((x_00, x_01, self.trcv_11(x_11)), dim = 1) )

        x_30 = self.conv_20_30( self.pool(x_20) )
        x_21 = self.conv_21( torch.cat((x_20, self.trcv_30(x_30)), dim = 1) )
        x_12 = self.conv_12( torch.cat((x_10, x_11, self.trcv_21(x_21)), dim = 1) )
        x_03 = self.conv_03( torch.cat((x_00, x_01, x_02, self.trcv_12(x_12)), dim = 1) )

        x_40 = self.conv_30_40( self.pool(x_30) )
        x_31 = self.conv_31( torch.cat((x_30, self.trcv_40(x_40)), dim = 1) )
        x_22 = self.conv_22( torch.cat((x_20, x_21, self.trcv_31(x_31)), dim = 1) )
        x_13 = self.conv_13( torch.cat((x_10, x_11, x_12, self.trcv_22(x_22)), dim = 1) )
        x_04 = self.conv_04( torch.cat((x_00, x_01, x_02, x_03, self.trcv_13(x_13)), dim = 1) )

        return self.activation(self.conv_out(x_04))

########################################################################################################################

class Panopticon_U3P(torch.nn.Module):
    """
    # Panopticon, Unet3+ architecture

    Using the Unet3+ architecture, identifies features and returns them on a 1:1 output map.
    """

    _NAME = "Panopticon_U3P"
    DEPTH = 5

    def __init__(self, internals: dict) -> None:
        """
        # Panopticon (Unet3+) initialization

        - `internals`: the `dict` with the internal parameters of the model (obtained using `model.export_values()`)

        If `internals` is not specified, the model will be created using the default values defined in `__init__()`.
        """
        super(Panopticon_U3P, self).__init__()

        # Saving the internals
        self.internals = internals

        # Basic shape
        self.channels_in = internals["general"]["channels_in"]
        self.channels_out = internals["general"]["channels_out"]
        self.init_features = internals["general"]["init_features"]
        self.drop_rate = internals["general"]["drop_rate"]

        # Defining parameters
        self.param_conv = internals["convolution"]
        self.param_trans = internals["trans_conv"]

        ##########
        # Creating levels

        init_l = torch.log2(torch.tensor(self.init_features))                                   # Computing the log2 of init feature
        num_features = 2**torch.arange(init_l, init_l + self.DEPTH, 1, dtype = int)             # Number of features at each depth

        up_channels = num_features[0] * self.DEPTH                                              # init_features * depth

        # Main nodes down
        self.conv_enc_i_0 = Block(self.channels_in, num_features[0], self.drop_rate, self.param_conv)           # Entry to 0,0
        self.conv_enc_0_1 = Block(num_features[0], num_features[1], self.drop_rate, self.param_conv)            # 0,0 to 1,0
        self.conv_enc_1_2 = Block(num_features[1], num_features[2], self.drop_rate, self.param_conv)            # 1,0 to 2,0
        self.conv_enc_2_3 = Block(num_features[2], num_features[3], self.drop_rate, self.param_conv)            # 2,0 to 3,0
        self.conv_enc_3_4 = Block(num_features[3], num_features[4], self.drop_rate, self.param_conv)            # 3,0 to 4,0

        # d3
        self.enc0_to_dec1 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,1
        self.enc1_to_dec_1 = BlockSingle(num_features[1], num_features[0], self.drop_rate, self.param_conv)     # 1,enc to dec,1
        self.enc2_to_dec1 = BlockSingle(num_features[2], num_features[0], self.drop_rate, self.param_conv)      # 2,enc to dec,1
        self.enc3_to_dec1 = BlockSingle(num_features[3], num_features[0], self.drop_rate, self.param_conv)      # 3,enc to dec,1
        self.enc4_to_dec1 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,1
        self.dec1 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,1

        # d2
        self.enc0_to_dec2 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,2
        self.enc1_to_dec2 = BlockSingle(num_features[1], num_features[0], self.drop_rate, self.param_conv)      # 1,enc to dec,2
        self.enc2_to_dec2 = BlockSingle(num_features[2], num_features[0], self.drop_rate, self.param_conv)      # 2,enc to dec,2
        self.enc3_to_dec2 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 3,enc to dec,2
        self.enc4_to_dec2 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,2
        self.dec2 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,2

        # d1
        self.enc0_to_dec3 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,3
        self.enc1_to_dec3 = BlockSingle(num_features[1], num_features[0], self.drop_rate, self.param_conv)      # 1,enc to dec,3
        self.enc2_to_dec3 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 2,enc to dec,3
        self.enc3_to_dec3 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 3,enc to dec,3
        self.enc4_to_dec3 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,3
        self.dec3 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,3

        # d0
        self.enc0_to_dec4 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,4
        self.enc1_to_dec4 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 1,enc to dec,4
        self.enc2_to_dec4 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 2,enc to dec,4
        self.enc3_to_dec4 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 3,enc to dec,4
        self.enc4_to_dec4 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,4
        self.dec4 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,4

        # Max pooling layer and up-sampling, static values so defined only once
        self.pool_2 = torch.nn.MaxPool1d(kernel_size = 2, stride = 2, ceil_mode = True)         # Pooling k=2
        self.pool_4 = torch.nn.MaxPool1d(kernel_size = 4, stride = 4, ceil_mode = True)         # Pooling k=4
        self.pool_8 = torch.nn.MaxPool1d(kernel_size = 8, stride = 8, ceil_mode = True)         # Pooling k=8
        self.up_2 = torch.nn.Upsample(scale_factor = 2, mode = "linear")                        # Up-sampling s=2
        self.up_4 = torch.nn.Upsample(scale_factor = 4, mode = "linear")                        # Up-sampling s=4
        self.up_8 = torch.nn.Upsample(scale_factor = 8, mode = "linear")                        # Up-sampling s=8
        self.up_16 = torch.nn.Upsample(scale_factor = 16, mode = "linear")                      # Up-sampling s=16

        # Convolution for output
        self.conv_out = torch.nn.Conv1d(up_channels, self.channels_out, 1)                      # dec,Last to out

        self.activation = _select_activation(internals["activation"])                           # Activation layer

    ########################################

    def forward(self, x: Tensor) -> Tensor:
        """
        # Forward pass at every call

        Note (from `torch` documentation): this method shouldn't be called directly, but prediction should
        use the module call directly, e.g. `Panopticon_U3P(x)`, as it ensures all registered hooks are ran properly.

        `x`: the input to evaluate, shape `[B, F, P]` (see module docs for details)

        Returns the class probability for each point.
        """

        # Computing backbone
        x_e0 = self.conv_enc_i_0(x)
        x_e1 = self.conv_enc_0_1( self.pool_2(x_e0) )
        x_e2 = self.conv_enc_1_2( self.pool_2(x_e1) )
        x_e3 = self.conv_enc_2_3( self.pool_2(x_e2) )
        x_d4 = self.conv_enc_3_4( self.pool_2(x_e3) )

        # Linking to d3
        x_e0_d3 = self.enc0_to_dec1( self.pool_8( x_e0 ) )
        x_e1_d3 = self.enc1_to_dec_1( self.pool_4( x_e1 ) )
        x_e2_d3 = self.enc2_to_dec1( self.pool_2( x_e2 ) )
        x_e3_d3 = self.enc3_to_dec1( x_e3 )
        x_e4_d3 = self.enc4_to_dec1( self.up_2( x_d4 ) )
        x_d3 = self.dec1( torch.cat( (x_e0_d3, x_e1_d3, x_e2_d3, x_e3_d3, x_e4_d3), dim = 1) )

        # Linking to d2
        x_e0_d2 = self.enc0_to_dec2( self.pool_4( x_e0 ) )
        x_e1_d2 = self.enc1_to_dec2( self.pool_2( x_e1 ) )
        x_e2_d2 = self.enc2_to_dec2( x_e2 )
        x_d3_d2 = self.enc3_to_dec2( self.up_2( x_d3 ) )
        x_d4_d2 = self.enc4_to_dec2( self.up_4( x_d4 ) )
        x_d2 = self.dec2( torch.cat( (x_e0_d2, x_e1_d2, x_e2_d2, x_d3_d2, x_d4_d2), dim = 1) )

        # Linking to d1
        x_e0_d1 = self.enc0_to_dec3( self.pool_2( x_e0 ) )
        x_e1_d1 = self.enc1_to_dec3( x_e1 )
        x_d2_d1 = self.enc2_to_dec3( self.up_2( x_d2 ) )
        x_d3_d1 = self.enc3_to_dec3( self.up_4( x_d3 ) )
        x_d4_d1 = self.enc4_to_dec3( self.up_8( x_d4 ) )
        x_d1 = self.dec3( torch.cat( (x_e0_d1, x_e1_d1, x_d2_d1, x_d3_d1, x_d4_d1), dim = 1) )

        # Linking to d0
        x_e0_d0 = self.enc0_to_dec4(x_e0)
        x_d1_d0 = self.enc1_to_dec4( self.up_2( x_d1 ) )
        x_d2_d0 = self.enc2_to_dec4( self.up_4( x_d2 ) )
        x_d3_d0 = self.enc3_to_dec4( self.up_8( x_d3 ) )
        x_d4_d0 = self.enc4_to_dec4( self.up_16( x_d4 ) )
        x_d0 = self.dec4( torch.cat( (x_e0_d0, x_d1_d0, x_d2_d0, x_d3_d0, x_d4_d0), dim = 1) )

        x_out = self.conv_out(x_d0)

        return self.activation( x_out )

########################################################################################################################

class Panopticon_U3P_PINN(torch.nn.Module):
    """
    # Panopticon, Unet3+ architecture

    Using the Unet3+ architecture, identifies features and returns them on a 1:1 output map.
    """

    _NAME = "Panopticon_U3P_PINN"
    DEPTH = 5

    def __init__(self, internals: dict) -> None:
        """
        # Panopticon (Unet3+) initialization

        - `internals`: the `dict` with the internal parameters of the model (obtained using `model.export_values()`)

        If `internals` is not specified, the model will be created using the default values defined in `__init__()`.
        """
        super(Panopticon_U3P, self).__init__()

        # Saving the internals
        self.internals = internals

        # Basic shape
        self.channels_in = internals["general"]["channels_in"]
        self.channels_out = internals["general"]["channels_out"]
        self.init_features = internals["general"]["init_features"]
        self.drop_rate = internals["general"]["drop_rate"]

        # Defining parameters
        self.param_conv = internals["convolution"]
        self.param_trans = internals["trans_conv"]

        ##########
        # Creating levels

        init_l = torch.log2(torch.tensor(self.init_features))                                   # Computing the log2 of init feature
        num_features = 2**torch.arange(init_l, init_l + self.DEPTH, 1, dtype = int)             # Number of features at each depth

        up_channels = num_features[0] * self.DEPTH                                              # init_features * depth

        # Main nodes down
        self.conv_enc_i_0 = Block(self.channels_in, num_features[0], self.drop_rate, self.param_conv)           # Entry to 0,0
        self.conv_enc_0_1 = Block(num_features[0], num_features[1], self.drop_rate, self.param_conv)            # 0,0 to 1,0
        self.conv_enc_1_2 = Block(num_features[1], num_features[2], self.drop_rate, self.param_conv)            # 1,0 to 2,0
        self.conv_enc_2_3 = Block(num_features[2], num_features[3], self.drop_rate, self.param_conv)            # 2,0 to 3,0
        self.conv_enc_3_4 = Block(num_features[3], num_features[4], self.drop_rate, self.param_conv)            # 3,0 to 4,0

        # d3
        self.enc0_to_dec1 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,1
        self.enc1_to_dec_1 = BlockSingle(num_features[1], num_features[0], self.drop_rate, self.param_conv)     # 1,enc to dec,1
        self.enc2_to_dec1 = BlockSingle(num_features[2], num_features[0], self.drop_rate, self.param_conv)      # 2,enc to dec,1
        self.enc3_to_dec1 = BlockSingle(num_features[3], num_features[0], self.drop_rate, self.param_conv)      # 3,enc to dec,1
        self.enc4_to_dec1 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,1
        self.dec1 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,1

        # d2
        self.enc0_to_dec2 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,2
        self.enc1_to_dec2 = BlockSingle(num_features[1], num_features[0], self.drop_rate, self.param_conv)      # 1,enc to dec,2
        self.enc2_to_dec2 = BlockSingle(num_features[2], num_features[0], self.drop_rate, self.param_conv)      # 2,enc to dec,2
        self.enc3_to_dec2 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 3,enc to dec,2
        self.enc4_to_dec2 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,2
        self.dec2 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,2

        # d1
        self.enc0_to_dec3 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,3
        self.enc1_to_dec3 = BlockSingle(num_features[1], num_features[0], self.drop_rate, self.param_conv)      # 1,enc to dec,3
        self.enc2_to_dec3 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 2,enc to dec,3
        self.enc3_to_dec3 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 3,enc to dec,3
        self.enc4_to_dec3 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,3
        self.dec3 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,3

        # d0
        self.enc0_to_dec4 = BlockSingle(num_features[0], num_features[0], self.drop_rate, self.param_conv)      # 0,enc to dec,4
        self.enc1_to_dec4 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 1,enc to dec,4
        self.enc2_to_dec4 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 2,enc to dec,4
        self.enc3_to_dec4 = BlockSingle(up_channels, num_features[0], self.drop_rate, self.param_conv)          # 3,enc to dec,4
        self.enc4_to_dec4 = BlockSingle(num_features[4], num_features[0], self.drop_rate, self.param_conv)      # 4,enc to dec,4
        self.dec4 = BlockSingle(up_channels, up_channels, self.drop_rate, self.param_conv)                      # dec,4

        # Max pooling layer and up-sampling, static values so defined only once
        self.pool_2 = torch.nn.MaxPool1d(kernel_size = 2, stride = 2, ceil_mode = True)         # Pooling k=2
        self.pool_4 = torch.nn.MaxPool1d(kernel_size = 4, stride = 4, ceil_mode = True)         # Pooling k=4
        self.pool_8 = torch.nn.MaxPool1d(kernel_size = 8, stride = 8, ceil_mode = True)         # Pooling k=8
        self.up_2 = torch.nn.Upsample(scale_factor = 2, mode = "linear")                        # Up-sampling s=2
        self.up_4 = torch.nn.Upsample(scale_factor = 4, mode = "linear")                        # Up-sampling s=4
        self.up_8 = torch.nn.Upsample(scale_factor = 8, mode = "linear")                        # Up-sampling s=8
        self.up_16 = torch.nn.Upsample(scale_factor = 16, mode = "linear")                      # Up-sampling s=16

        # Convolution for output
        self.conv_out = torch.nn.Conv1d(up_channels, self.channels_out, 1)                      # dec,Last to out

        self.activation = _select_activation(internals["activation"])                           # Activation layer

    ########################################

    def forward(self, x: Tensor) -> Tensor:
        """
        # Forward pass at every call

        Note (from `torch` documentation): this method shouldn't be called directly, but prediction should
        use the module call directly, e.g. `Panopticon_U3P(x)`, as it ensures all registered hooks are ran properly.

        `x`: the input to evaluate, shape `[B, F, P]` (see module docs for details)

        Returns the class probability for each point.
        """

        # Computing backbone
        x_e0 = self.conv_enc_i_0(x)
        x_e1 = self.conv_enc_0_1( self.pool_2(x_e0) )
        x_e2 = self.conv_enc_1_2( self.pool_2(x_e1) )
        x_e3 = self.conv_enc_2_3( self.pool_2(x_e2) )
        x_d4 = self.conv_enc_3_4( self.pool_2(x_e3) )

        # Linking to d3
        x_e0_d3 = self.enc0_to_dec1( self.pool_8( x_e0 ) )
        x_e1_d3 = self.enc1_to_dec_1( self.pool_4( x_e1 ) )
        x_e2_d3 = self.enc2_to_dec1( self.pool_2( x_e2 ) )
        x_e3_d3 = self.enc3_to_dec1( x_e3 )
        x_e4_d3 = self.enc4_to_dec1( self.up_2( x_d4 ) )
        x_d3 = self.dec1( torch.cat( (x_e0_d3, x_e1_d3, x_e2_d3, x_e3_d3, x_e4_d3), dim = 1) )

        # Linking to d2
        x_e0_d2 = self.enc0_to_dec2( self.pool_4( x_e0 ) )
        x_e1_d2 = self.enc1_to_dec2( self.pool_2( x_e1 ) )
        x_e2_d2 = self.enc2_to_dec2( x_e2 )
        x_d3_d2 = self.enc3_to_dec2( self.up_2( x_d3 ) )
        x_d4_d2 = self.enc4_to_dec2( self.up_4( x_d4 ) )
        x_d2 = self.dec2( torch.cat( (x_e0_d2, x_e1_d2, x_e2_d2, x_d3_d2, x_d4_d2), dim = 1) )

        # Linking to d1
        x_e0_d1 = self.enc0_to_dec3( self.pool_2( x_e0 ) )
        x_e1_d1 = self.enc1_to_dec3( x_e1 )
        x_d2_d1 = self.enc2_to_dec3( self.up_2( x_d2 ) )
        x_d3_d1 = self.enc3_to_dec3( self.up_4( x_d3 ) )
        x_d4_d1 = self.enc4_to_dec3( self.up_8( x_d4 ) )
        x_d1 = self.dec3( torch.cat( (x_e0_d1, x_e1_d1, x_d2_d1, x_d3_d1, x_d4_d1), dim = 1) )

        # Linking to d0
        x_e0_d0 = self.enc0_to_dec4(x_e0)
        x_d1_d0 = self.enc1_to_dec4( self.up_2( x_d1 ) )
        x_d2_d0 = self.enc2_to_dec4( self.up_4( x_d2 ) )
        x_d3_d0 = self.enc3_to_dec4( self.up_8( x_d3 ) )
        x_d4_d0 = self.enc4_to_dec4( self.up_16( x_d4 ) )
        x_d0 = self.dec4( torch.cat( (x_e0_d0, x_d1_d0, x_d2_d0, x_d3_d0, x_d4_d0), dim = 1) )

        x_out = self.conv_out(x_d0)

        return self.activation( x_out )

########################################################################################################################

class Block(torch.nn.Module):
    """
    # Convolution blocks for Unets

    This block applies two convolution in a row of the same shape.
    """

    def __init__(self, in_channels: int, features: int, drop_rate: float, param_conv: dict) -> None:
        """
        # Block initialization

        `in_channels`: the number of input channels of the block
        `features`: the number of output channels of the block
        `param_conv`: the required parameters for the convolution sub-blocks
        """
        super(Block, self).__init__()

        self.relu = torch.nn.ReLU()                                     # ReLU function, not trainable
        self.conv_entry = torch.nn.Conv1d(                              # The first conv
            in_channels = in_channels,                                  # Num channel is the input
            out_channels = features,                                    # We output the desired num features
            **param_conv                                                # Rest of the parameters
        )
        self.norm_1 = torch.nn.BatchNorm1d(num_features = features)     # First BatchNorm

        self.conv_enforce = torch.nn.Conv1d(                            # The second convolution
            in_channels = features,                                     # In and Out have the same size
            out_channels = features,                                    # Only used to reinforce the features
            **param_conv                                                # Rest of the parameters
        )
        self.drop = torch.nn.Dropout1d(p = drop_rate)                   # Dropout chance
        self.norm_2 = torch.nn.BatchNorm1d(num_features = features)     # Second BatchNorm
    
    ########################################

    def forward(self, x: Tensor) -> Tensor:
        """
        # Forward pass at every call

        Note (from `torch` documentation): this method shouldn't be called directly, but prediction should
        use the module call directly, e.g. `Block(x)`, as it ensures all registered hooks are ran properly.

        `x`: the input to evaluate, shape `[B, F, P]` (see module docs for details)

        Returns the double convolution block of the input.
        """

        x = self.drop(x)                # Dropout ->
        x = self.conv_entry(x)          # First Conv ->
        x = self.relu(self.norm_1(x))   # BatchNorm1 -> ReLU ->
        x = self.conv_enforce(x)        # Second Conv ->
        x = self.relu(self.norm_2(x))   # BatchNorm2 -> ReLU ->

        return x                        # Output

########################################

class BlockSingle(torch.nn.Module):
    """
    # Convolution blocks for Unets, single conv
    """

    def __init__(self, in_channels: int, features: int, drop_rate: float, param_conv: dict) -> None:
        super(BlockSingle, self).__init__()

        self.relu = torch.nn.ReLU()                                     # ReLU function, not trainable

        self.conv_entry = torch.nn.Conv1d(                              # The actual convolution
            in_channels = in_channels,                                  # Num channel is the input
            out_channels = features,                                    # We output the desired num features
            **param_conv                                                # Rest of the parameters
        )
        self.norm = torch.nn.BatchNorm1d(num_features = features)       # BatchNorm
        self.drop = torch.nn.Dropout1d(p = drop_rate)                   # Dropout chance

    def forward(self, x: Tensor) -> Tensor:
        """
        # Forward pass at every call

        Note (from `torch` documentation): this method shouldn't be called directly, but prediction should
        use the module call directly, e.g. `BlockSingle(x)`, as it ensures all registered hooks are ran properly.

        `x`: the input to evaluate, shape `[B, F, P]` (see module docs for details)

        Returns the convolution block of the input.
        """

        x = self.drop(x)                # Dropout ->
        x = self.conv_entry(x)          # First Conv ->
        x = self.relu(self.norm(x))     # BatchNorm -> ReLU

        return x

########################################################################################################################
