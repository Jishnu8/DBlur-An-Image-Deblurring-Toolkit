from dblur.testers.base_tester import BaseTester
from dblur.models.textdbn import TextDBN
import torch
import torch.nn as nn


class TextDBNTester(BaseTester):
    """TextDBN tester for image deblurring. 

    TextDBNTester is subclassed from BaseTester. The class contains methods to 
    test a model used for deblurring, deblur multiple/single images given a 
    pretrained model, get test dataloader get model, and get loss function."""

    def get_model(self, backbone_name="L15", list_of_filter_sizes=[23, 1, 1, 1, 1, 3, 1, 5, 3, 5],
                  list_of_no_of_channels=[128, 320, 320, 320, 128, 128, 512, 48, 96, 3]):
        """Returns an instance of TextDBN model given the arguments below:

        Args:
            backbone_name: specifies a backbone name which is referred to in the
                paper "convolutional neural networks for direct text 
                deblurring". Possible backbones are: "L10S", "L10M", "L10L", 
                "L15", "custom". For "custom", the following arguments must be 
                specified. 
            list_of_filter_sizes: list of kernel sizes for the sequence of 
                CNNBlock that makes up the model.
            list_of_no_of_channels: list of output channels for the sequence of
                CNNBlock that makes up the model. The last element has to 3 to 
                obtain a valid RGB image as the output.
        """

        model = TextDBN(backbone_name, list_of_filter_sizes, list_of_no_of_channels)
        return model

    def get_loss(self):
        """Returns Mean Square Error Loss used for testing TextDBN."""

        return nn.MSELoss()
