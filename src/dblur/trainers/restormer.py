from dblur.trainers.base_trainer import BaseTrainer
from dblur.models.restormer import Restormer
from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


class RestormerTrainer(BaseTrainer):
    """Restormer trainer for image deblurring. 
    
    RestormerTrainer is subclassed from BaseTrainer. It contains methods to 
    train, validate, get model, get loss function, get optimizer, 
    get learning rate scheduler, get train dataloader and get validation 
    dataloader. The default arguments in each method has been given a value as 
    specified in the paper "Restormer: Efficient Transformer for High-Resolution 
    Image Restoration" but can be changed if required. For more details 
    regarding the arguments in each method, refer to the paper. 
    """

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
        """Returns L1Loss which is used for training Restormer"""

        return nn.L1Loss()

    def get_optimizer(self, params, learning_rate=3e-4, betas=(0.9, 0.999), weight_decay=0):
        """Returns Adam optimizer used for training Restormer. 

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """

        return torch.optim.AdamW(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer, T_max=1000, eta_min=1e-6):
        """Returns cosine annealing learning rate scheduler used for training Restormer."""

        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    def get_train_dataloader(self, dataset_path, transform="default", batch_size=16):
        """See base class."""

        if transform == "default":
            train_composed = transforms.Compose([RandomCrop(256), RandomHorizontalFlip(), RandomVerticalFlip()])
        else:
            train_composed = transform

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("Warning: Cuda not found. Training will occur on the cpu")

        train_dataset = ImageDataset(dataset_path, train_composed)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader

    def get_val_dataloader(self, dataset_path, transform="default", batch_size=16):
        """See base class."""

        if transform == "default":
            val_composed = transforms.Compose([RandomCrop(256)])
        else:
            val_composed = transform

        val_dataset = ImageDataset(dataset_path, val_composed)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return val_dataloader
