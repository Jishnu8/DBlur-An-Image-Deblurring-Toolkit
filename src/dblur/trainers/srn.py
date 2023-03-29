from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop
from dblur.trainers.base_trainer import BaseTrainer
from dblur.models.srn import SRN
from dblur.losses.srn import SRNLoss
import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class SRNTrainer(BaseTrainer):
    """SRN trainer for image deblurring. 
    
    SRNTrainer is subclassed from BaseTrainer. It contains methods to 
    train, validate, get model, get loss function, get optimizer, 
    get learning rate scheduler, get train dataloader and get validation 
    dataloader. The default arguments in each method has been given a value as 
    specified in the paper "Scale-recurrent Network for Deep Image Deblurring" 
    but can be changed if required. For more details regarding the arguments in 
    each method, refer to the paper. 
    """

    def validate(self, model, val_dataloader, loss_func, device, batch_count, writer=None):
        model.eval()
        running_loss = 0

        psnr = PeakSignalNoiseRatio().to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_score = 0
        ssim_score = 0

        with torch.no_grad():
            for X, y in val_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = loss_func(pred, y)
                running_loss += loss

                denormalized_pred = pred[len(pred) - 1] + 0.5  # difference here
                denormalized_y = y + 0.5
                denormalized_pred = torch.clamp(denormalized_pred, 0, 1)
                denormalized_y = torch.clamp(denormalized_y, 0, 1)

                psnr_score += psnr(denormalized_pred, denormalized_y)
                ssim_score += ssim(denormalized_pred, denormalized_y)

        print(f"\nval loss: {running_loss / len(val_dataloader):>7f}  [{batch_count:>3d}]")
        print(f"psnr scr: {psnr_score / len(val_dataloader):>7f}  [{batch_count:>3d}]")
        print(f"ssim scr: {ssim_score / len(val_dataloader):>7f}  [{batch_count:>3d}]")
        print("\n")
        # print('validation loss: %.5f [%d       ] ' % (batch_count, running_loss / len(val_dataloader)))

        if writer is not None:
            writer.add_scalar('val_loss', running_loss / len(val_dataloader), batch_count)
            writer.add_scalar('psnr_score', psnr_score / len(val_dataloader), batch_count)
            writer.add_scalar('ssim_score', ssim_score / len(val_dataloader), batch_count)

    def get_model(self, num_of_layers=3, num_of_e_blocks=2):
        """Returns an instance of the SRN model given the argument below:

        Args:
            num_of_layers: number of layers in SRN model
            num_of_e_blocks: number of encoder/decoder blocks in each layer
        """

        return SRN(num_of_layers=num_of_layers, num_of_e_blocks=num_of_e_blocks)

    def get_loss(self):
        """Returns loss function used for training SRN"""
        return SRNLoss()

    def get_optimizer(self, params, learning_rate=1e-4, betas=(0.9, 0.999), weight_decay=0):
        """Returns Adam optimizer used for training SRN. 

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """

        return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer, gamma=0.3):
        """Returns Exponential Learning Rate Scheduler used for training SRN."""

        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    def get_train_dataloader(self, dataset_path, transform="default", batch_size=16):
        """See base class."""

        if transform == "default":
            train_composed = transforms.Compose([RandomCrop(256)])
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
