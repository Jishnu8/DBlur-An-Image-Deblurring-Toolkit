from dblur.testers.base_tester import BaseTester
from dblur.data.dataset import ImageDataset
from dblur.models.nafnet import NAFNet
from dblur.losses.psnr_loss import PSNRLoss
import torch.nn as nn
import os
from torch.utils.data import DataLoader


class NAFNetTester(BaseTester):
    """NAFNet tester for image deblurring. 

    NAFNetTester is subclassed from BaseTester. The class contains methods to 
    test a model used for deblurring, deblur multiple/single images given a 
    pretrained model, get test dataloader, get model and get loss function."""

    def get_test_dataloader(self, dataset_path, transform=None, batch_size=32):
        """See base class."""

        test_dataset = ImageDataset(dataset_path, transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return test_dataloader

    def get_model(self, projected_in_channels=32, enc_num=[1, 1, 1, 28], middle_num=1, dec_num=[1, 1, 1, 1],
                  attn_expansion_factor=2, ffn_expansion_factor=2, gate_reduction_factor=2, dropout_rate=0,
                  upscale_factor=2, attn_kernel_size=3, upscale_kernel_size=1, bias=True, upscale_bias=False):

        """Returns an instance of the NAFNet model based on the arguments below:

        Args:
            projected_in_channels: number of channels input is projected into by
                the first convolution layer.
            enc_num: list of number of NAFNet blocks in each encoder for a 
                particular layer of the UNet. 
            middle_num: number of NAFNet blocks in lowest layer of the UNet. 
            dec_num: list of number of NAFNet blocks in each decoder for a 
                particular layer of the UNet. 
            attn_expansion_factor: expansion factor in the 1D Convolution in the 
                attention block of each NAFNet Block.
            ffn_expansion_factor: expansion factor in the 1D Convolution in the 
                Feed Forward Block of each NAFNet Block.
            gate_reduction_factor: reductor factor in the SimpleGate present in 
                each Feed Forward Block of each NAFNet Block.
            dropout_rate: dropout rate in each NAFNet Block.
            upscale_factor: scaling factor across different layers in the UNet. 
                E.g. With a scaling factor of 2, the image dimensions are halved 
                as we go move downward across layers in the UNet. 
            attn_kernel_size: kernel size of the convolutional layer in each 
                attention block in each NAFNet Block.
            upscale_kernel_size: kernel size of the convolution layer in each 
                UpSample and DownSample block.
            bias: if True, bias is added in every convolution layer in the 
                model except for the convolutions present in the UpSample and 
                DownSample blocks. 
            upscale_bias: if True, bias is added in every convolution layer in 
                the UpSample and Donwsample block.
        """

        return NAFNet(projected_in_channels=projected_in_channels, enc_num=enc_num, middle_num=middle_num,
                      dec_num=dec_num,
                      attn_expansion_factor=attn_expansion_factor, ffn_expansion_factor=ffn_expansion_factor,
                      gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate,
                      upscale_factor=upscale_factor,
                      attn_kernel_size=attn_kernel_size, upscale_kernel_size=upscale_kernel_size, bias=bias,
                      upscale_bias=upscale_bias)

    def get_loss(self, psnr=False):
        """Returns Loss which is used for testing NAFNet.

        If psnr == True, returns PSNR Loss. Alternatively, if psnr == false,
        returns MSE Loss
        """

        if psnr:
            return PSNRLoss()
        else:
            return nn.MSELoss()
