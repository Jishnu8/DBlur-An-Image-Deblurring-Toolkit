import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """CNN Block which consists of a convolution layer, followed by a batch norm layer and a ReLU activation.
    """

    def __init__(self, filter_size, in_channels, out_channels):
        """Constructor of CNNBlock.

        Args:
            filter_size: size of kernel for the convolutional layer.
            in_channels: number of channels of the input torch.Tensor.
            out_channels: number of channels of the output torch.Tensor.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride=1, padding="same", bias="True")
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace="false")

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class TextDBN(nn.Module):
    """TextDBN model for image deblurring. 
    
    The model consists of a sequence of CNNBlocks (convolution layer, 
    batch norm, ReLU). The arguments for each CNNBlock can be specified in the 
    constructor. Details regarding the model architecture can be found in the 
    paper "Convolutional neural networks for direct text deblurring". 
    """

    def __init__(self, backbone_name="L10S", list_of_filter_sizes=[23, 1, 1, 1, 1, 3, 1, 5, 3, 5],
                 list_of_no_of_channels=[128, 320, 320, 320, 128, 128, 512, 48, 96, 3]):
        """Constructor of TextDBN. Default values have set for each argument as per the paper.

        Args:
            backbone_name: specifies a backbone name which is referred to in the
                paper "convolutional neural networks for direct text 
                deblurring". Possible backbones are: "L10S", "L10M", "L10L", 
                "L15", "custom". For "custom", the following arguments must be 
                specified. 
            list_of_filter_sizes: list of kernel sizes for the sequence of 
                CNNBlock that makes up the model.
            list_of_no_channels: list of output channels for the sequence of 
                CNNBlock that makes up the model. The last element has to 3 to 
                obtain a valid RGB image as the ouput.  
        """

        super().__init__()
        if len(list_of_filter_sizes) != len(list_of_no_of_channels):
            raise Exception("number of filters sizes and number of channels are not the same.")

        if list_of_no_of_channels[len(list_of_no_of_channels) - 1] != 3:
            raise Exception("Number of channels for the last layer has to be 3.")

        if backbone_name == "L10S":
            self.list_of_filter_sizes = [23, 1, 1, 1, 1, 3, 1, 5, 3, 5]
            self.list_of_no_of_channels = [128, 320, 320, 320, 128, 128, 512, 48, 96, 3]
        elif backbone_name == "L10M":
            self.list_of_filter_sizes = [23, 1, 1, 1, 1, 3, 1, 5, 3, 5]
            self.list_of_no_of_channels = [194, 400, 400, 400, 156, 156, 512, 56, 128, 3]
        elif backbone_name == "L10L":
            self.list_of_filter_sizes = [23, 1, 1, 1, 1, 3, 1, 5, 3, 5]
            self.list_of_no_of_channels = [220, 512, 512, 512, 196, 196, 512, 64, 196, 3]
        elif backbone_name == "L15":
            self.list_of_filter_sizes = [19, 1, 1, 1, 1, 3, 1, 5, 5, 3, 5, 5, 1, 7, 7]
            self.list_of_no_of_channels = [128, 320, 320, 320, 128, 128, 512, 128, 128, 128, 128, 128, 256, 64, 3]
        elif backbone_name == "custom":
            self.list_of_filter_sizes = list_of_filter_sizes
            self.list_of_no_of_channels = list_of_no_of_channels
        else:
            raise Exception("Please enter a valid backbone_name. For a custom backbone_name, set backbone_name=custom")

        self.list_of_conv_blocks = nn.ModuleList()
        self.list_of_conv_blocks.append(CNNBlock(self.list_of_filter_sizes[0], 3, self.list_of_no_of_channels[0]))
        for i in range(1, len(self.list_of_filter_sizes) - 1):
            self.list_of_conv_blocks.append(CNNBlock(self.list_of_filter_sizes[i], self.list_of_no_of_channels[i - 1],
                                                     self.list_of_no_of_channels[i]))

        self.last_layer_conv = nn.Conv2d(self.list_of_no_of_channels[len(self.list_of_filter_sizes) - 2],
                                         self.list_of_no_of_channels[len(self.list_of_filter_sizes) - 1],
                                         self.list_of_filter_sizes[len(self.list_of_filter_sizes) - 1], stride=1,
                                         padding="same", bias="True")
        self.list_of_conv_blocks.append(self.last_layer_conv)

    def forward(self, x):
        for i in range(len(self.list_of_conv_blocks)):
            x = self.list_of_conv_blocks[i](x)

        return x
