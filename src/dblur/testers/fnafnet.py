from dblur.testers.base_tester import BaseTester
from dblur.data.dataset import ImageDataset
from dblur.models.fnafnet import FNAFNet
from dblur.losses.fnafnet import FNAFNetLoss
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader


class FNAFNetTester(BaseTester):
    """FNAFNet tester for image deblurring. 

    FNAFNetTester is subclassed from BaseTester. The class contains methods to 
    test a model used for deblurring, deblur multiple/single images given a 
    pretrained model, get test dataloader, get model and get loss function."""

    def get_test_dataloader(self, dataset_path, transform=None, batch_size=16):
        """See base class."""

        test_dataset = ImageDataset(dataset_path, transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return test_dataloader

    def get_model(self, projected_in_channels=32, enc_num=[1, 1, 1, 28], middle_num=1, dec_num=[1, 1, 1, 1],
                  attn_expansion_factor=2, ffn_expansion_factor=2, gate_reduction_factor=2, dropout_rate=0,
                  upscale_factor=2, attn_kernel_size=3, upscale_kernel_size=1, bias=True, upscale_bias=False,
                  fft_block_expansion_factor=2, fft_block_norm="backward", fft_block_activation=nn.ReLU(),
                  fft_block_bias=False, fft_block_a=4):
        """Returns an instance of the FNAFNet model based on the arguments below:
        
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
            bias: if True, bias is added in every convolution layer in the model
                except for the convolutions present in the UpSample and DownSample
                blocks.
            upscale_bias: if True, bias is added in every convolution layer in the
                UpSample and Donwsample block.
            fft_block_expansion_factor: expansion factor of the 1D convolution
                present in each ResFFTBlock.
            fft_block_norm: norm of the inverse 2D Fourier Transform present in
                each ResFFTBlock.
            fft_block_activation: activation function present in each ResFFTBlock.
            fft_block_bias: if True, bias is added in the 1D convolution present
                in each ResFFTBlock.
            fft_block_a: the negative slope of the rectifier used after this layer
                for the initialization of weights using
                torch.nn.init.kaiming_uniform.
        """

        return FNAFNet(projected_in_channels=projected_in_channels, enc_num=enc_num, middle_num=middle_num,
                       dec_num=dec_num,
                       attn_expansion_factor=attn_expansion_factor, ffn_expansion_factor=ffn_expansion_factor,
                       gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate,
                       upscale_factor=upscale_factor,
                       attn_kernel_size=attn_kernel_size, upscale_kernel_size=upscale_kernel_size, bias=bias,
                       upscale_bias=upscale_bias,
                       fft_block_expansion_factor=fft_block_expansion_factor, fft_block_norm=fft_block_norm,
                       fft_block_activation=fft_block_activation, fft_block_bias=fft_block_bias,
                       fft_block_a=fft_block_a)

    def get_loss(self):
        """Returns FAFNet Loss used for training FNAFNet."""

        return FNAFNetLoss()
