import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop, RandomHorizontalFlip, RotationTransform
from dblur.trainers.base_trainer import BaseTrainer
from dblur.models.textdbn import TextDBN
from dblur.schedulers.textdbn import TextDBNScheduler


class TextDBNTrainer(BaseTrainer):
    """TextDBN trainer for image deblurring. 
    
    TextDBNTrainer is subclassed from BaseTrainer. It contains methods to train,
    validate, get model, get loss function, get optimizer, get learning rate 
    scheduler, get train dataloader and get validation dataloader. The default
    arguments in each method has been given a value as specified in the paper 
    "Convolutional neural networks for direct text deblurring" but can be 
    changed if required. For more details regarding the arguments in each 
    method, refer to the paper. 
    """

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
        """Returns Mean Square Error Loss used for training TextDBN."""

        return nn.MSELoss()

    def get_optimizer(self, params, learning_rate=1e-4, betas=(0.9, 0.999), weight_decay=1e-5):
        """Returns Adam optimizer used for training TextDBN
         specified by the arguments below:

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """

        return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer):
        """Returns learning rate scheduler specified in paper used for training TextDBN"""

        return TextDBNScheduler(optimizer)

    def get_train_dataloader(self, dataset_path, transform="default", batch_size=8):
        """See base class"""

        if transform == "default":
            train_composed = transforms.Compose(
                [RandomCrop(256), RandomHorizontalFlip(), RotationTransform(angles=[0, 90, 180, 270])])
        else:
            train_composed = transform

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("Warning: Cuda not found. Training will occur on the cpu")

        train_dataset = ImageDataset(dataset_path, train_composed)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader

    def get_val_dataloader(self, dataset_path, transform="default", batch_size=8):
        """See base class"""

        if transform == "default":
            val_composed = transforms.Compose([RandomCrop(160)])
        else:
            val_composed = transform

        val_dataset = ImageDataset(dataset_path, val_composed)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return val_dataloader
