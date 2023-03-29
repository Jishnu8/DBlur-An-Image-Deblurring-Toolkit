from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop
from dblur.trainers.base_trainer import BaseTrainer
from dblur.models.stack_dmphn import StackDMPHN, Encoder, Decoder
from dblur.losses.stack_dmphn import StackDMPHNLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import DataLoader


class StackDMPHNTrainer(BaseTrainer):
    """StackDMPHN trainer for image deblurring. 
    
    StackDMPHNTrainer is subclassed from BaseTrainer. It contains methods to 
    train, validate, get model, get loss function, get optimizer, 
    get learning rate scheduler, get train dataloader and get validation 
    dataloader. The default arguments in each method has been given a value as 
    specified in the paper "Deep Stacked Hierarchical Multi-patch Network for 
    Image Deblurring" but can be changed if required. For more details regarding 
    the arguments in each method, refer to the paper. 
    """

    def validate(self, model, val_dataloader, loss_func, device, batch_count, writer=None):
        """See base class"""

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

                denormalized_pred = pred[len(pred) - 1] + 0.5
                denormalized_y = y + 0.5
                denormalized_pred = torch.clamp(denormalized_pred, 0, 1)
                denormalized_y = torch.clamp(denormalized_y, 0, 1)

                psnr_score += psnr(denormalized_pred, denormalized_y)
                ssim_score += ssim(denormalized_pred, denormalized_y)

        print(f"\nval loss: {running_loss / len(val_dataloader):>7f}  [{batch_count:>3d}]")
        print(f"psnr scr: {psnr_score / len(val_dataloader):>7f}  [{batch_count:>3d}]")
        print(f"ssim scr: {ssim_score / len(val_dataloader):>7f}  [{batch_count:>3d}]")
        print("\n")

        if writer is not None:
            writer.add_scalar('val_loss', running_loss / len(val_dataloader), batch_count)
            writer.add_scalar('psnr_score', psnr_score / len(val_dataloader), batch_count)
            writer.add_scalar('ssim_score', ssim_score / len(val_dataloader), batch_count)

    def get_model(self, num_of_stacks=4, encoder=Encoder(), decoder=Decoder(), num_layers=4):
        """Returns an instance of StackDMPHN given the arguments below: 
        
        Args:
            num_of_stacks: number of stacks in each DMPHN module.
            encoder: encoder used in each layer of a DMPHN module.
            decoder: decoded used in each layer of a DMPHN module.
            num_layers: number of DMPHN modules.
        """

        return StackDMPHN(num_of_stacks=num_of_stacks, encoder=encoder, decoder=decoder, num_layers=num_layers)

    def get_loss(self):
        """Returns loss function used for training StackDMPHN model"""

        return StackDMPHNLoss()

    def get_optimizer(self, params, learning_rate=1e-4, betas=(0.9, 0.999), weight_decay=0):
        """Returns Adam optimizer used for training StackDMPHN. 

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """

        return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer, step_size=1000, gamma=0.1):
        """Returns learning rate scheduler used for training StackDMPHN."""

        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    def get_train_dataloader(self, dataset_path, transform="default", batch_size=6):
        """See base class"""

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

    def get_val_dataloader(self, dataset_path, transform="default", batch_size=6):
        """See base class"""

        if transform == "default":
            val_composed = transforms.Compose([RandomCrop(160)])
        else:
            val_composed = transform

        val_dataset = ImageDataset(dataset_path, val_composed)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return val_dataloader
