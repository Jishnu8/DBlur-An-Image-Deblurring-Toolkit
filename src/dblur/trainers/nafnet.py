import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop, RandomHorizontalFlip, RotationTransform, RandomVerticalFlip
from dblur.trainers.base_trainer import BaseTrainer
from dblur.models.nafnet import NAFNet
from dblur.losses.psnr_loss import PSNRLoss


class NAFNetTrainer(BaseTrainer):
    """NAFNet trainer for image deblurring. 
    
    NAFNetTrainer is subclassed from BaseTrainer. It contains methods to 
    train, validate, get model, get loss function, get optimizer, 
    get learning rate scheduler, get train dataloader and get validation 
    dataloader. The default arguments in each method has been given a value as 
    specified in the paper "Simple Baselines for Image Restoration" but can be 
    changed if required. For more details regarding the arguments in each 
    method, refer to the paper. 
    """

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
        """Returns Loss which is used for training NAFNet.

        If psnr == True, returns PSNR Loss. Alternatively, if psnr == false,
        returns MSE Loss
        """

        if psnr:
            return PSNRLoss()
        else:
            return nn.MSELoss()

    def get_optimizer(self, params, learning_rate=1e-3, betas=(0.9, 0.9), weight_decay=0):
        """Returns Adam optimizer used for training NAFNet. 

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """

        return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer, T_max=1000, eta_min=1e-6):
        """Return Consine Annealing Learning Rate Scheduler used for training 
        NAFNet."""

        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    def get_train_dataloader(self, dataset_path, transform="default", batch_size=32):
        """See base class."""

        if transform == "default":
            train_composed = transforms.Compose(
                [RandomCrop(256), RandomHorizontalFlip(), RandomVerticalFlip(), RotationTransform([0, 90, 180, 270])])
        else:
            train_composed = transform

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("Warning: Cuda not found. Training will occur on the cpu")

        train_dataset = ImageDataset(dataset_path, train_composed)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader

    def get_val_dataloader(self, dataset_path, transform="default", batch_size=32):
        """See base class."""

        if transform == "default":
            val_composed = transforms.Compose([RandomCrop(256)])
        else:
            val_composed = transform

        val_dataset = ImageDataset(dataset_path, val_composed)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return val_dataloader
