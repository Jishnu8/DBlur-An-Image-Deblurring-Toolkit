from dblur.testers.base_tester import BaseTester
from dblur.models.restormer import Restormer
import torch.nn as nn


class RestormerTester(BaseTester):
    """Restormer tester for image deblurring. 

    RestormerTester is subclassed from BaseTester. The class contains methods to 
    test a model used for deblurring, deblur multiple/single images given a 
    pretrained model, get test dataloader, get model and get loss function."""

    def get_model(self, num_layers=4, num_transformer_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
                  channels=[48, 96, 192, 384], num_refinement_blocks=4, expansion_factor=2.66, bias=False):
        """Returns an instance of Restormer given the arguments below:
        
        Args:
          num_layers: number of layers in Restormer.
          num_transformer_blocks: list containing number of transformer blocks
              in each layer for both the encoder and decoder .
          heads: list of number of attention heads in each layer of both the 
              encoder and decoder.
          channels: list of number of input channels in input for each layer of 
              both the encoder and decoder
          num_refinement_blocks: number of refinement transformer blocks used 
          expansion_factor: expansion factor GDFN module of each Transformer 
              block. 
          bias: if True, each convolutional layer will have a bias in model.
        """

        return Restormer(num_layers=num_layers, num_transformer_blocks=num_transformer_blocks, heads=heads,
                         channels=channels,
                         num_refinement_blocks=num_refinement_blocks, expansion_factor=expansion_factor, bias=bias)

    def get_loss(self):
        """Returns L1Loss which is used for testing Restormer"""

        return nn.L1Loss()
