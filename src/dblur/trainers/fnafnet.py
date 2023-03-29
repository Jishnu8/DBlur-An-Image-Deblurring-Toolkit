import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from dblur.trainers.base_trainer import BaseTrainer
from dblur.models.fnafnet import FNAFNet
from dblur.losses.fnafnet import FNAFNetLoss
from torch.utils.tensorboard import SummaryWriter
import os


class FNAFNetTrainer(BaseTrainer):
    """FNAFNet trainer for image deblurring. 
    
    FNAFNetTrainer is subclassed from BaseTrainer. It contains methods to 
    train, validate, get model, get loss function, get optimizer, 
    get learning rate scheduler, get train dataloader and get validation 
    dataloader. The default arguments in each method has been given a value as 
    specified in the paper "Intriguing Findings of Frequency Selection for Image 
    Deblurring" but can be changed if required. For more details regarding the 
    arguments in each method, refer to the paper. 
    """

    def train(
            self,
            model=None,
            train_dataloader=None,
            val_dataloader=None,
            optimizer=None,
            loss_func=None,
            lr_scheduler=None,
            scheduler_step_every_batch=False,
            epochs=2000,
            checkpoint_save_name=None,
            save_checkpoint_freq=300,
            val_freq=100,
            write_logs=False,
            logs_folder='runs',
    ):

        """See base class."""

        if checkpoint_save_name is None:
            raise Exception("Path to directory for saving checkpoint was not provided.")
        if train_dataloader is None:
            raise Exception("train_dataloader was not provided.")
        if model is None and optimizer is None:
            print("Loading default model specified in get_model() as no model was provided")
            model = self.get_model()
            print("Loading default optimizer specified in get_optimizer() as no optimizer was provided")
            optimizer = self.get_optimizer(model.parameters())
        if loss_func is None:
            print("Loading default loss_func specified in get_loss() as no loss was provided")
            optimizer = self.get_loss()
        if val_dataloader is None:
            print("Warning. val_dataloader was not provided.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if write_logs:
            writer = SummaryWriter(logs_folder)
        else:
            writer = None

        model = model.to(device)
        if os.path.exists(checkpoint_save_name):
            checkpoint = torch.load(checkpoint_save_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print("Existing model loaded")

        model.train()

        seq_count = 0
        batch_size = train_dataloader.batch_size
        size = len(train_dataloader.dataset)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            epoch_loss = 0
            for g in optimizer.param_groups:
                print("Learning rate: ", g['lr'])
            seq_batch_count = 0
            for train_batch, (X, y) in enumerate(train_dataloader):
                X = X.to(device)
                y = y.to(device)

                pred = model(X)
                loss = loss_func(pred, y)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                seq_count += 1
                seq_batch_count += 1
                current = (train_batch + 1) * batch_size
                if current > size:
                    current = size
                print(f"loss: {loss.detach().data:>7f}  [{current:>5d}/{size:>5d}]")

                epoch_loss += loss.item()
                if write_logs:
                    writer.add_scalar("train_loss", float(loss.detach().data), seq_count)

                if seq_count % save_checkpoint_freq == 0:
                    if lr_scheduler is not None:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict()
                        }, checkpoint_save_name)
                    else:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_save_name)

                if seq_count % val_freq == 0:
                    self.validate(model, val_dataloader, loss_func, device, seq_count // val_freq, writer)
                    model.train()

                if lr_scheduler is not None and scheduler_step_every_batch is True:
                    lr_scheduler.step()

            if lr_scheduler is not None and scheduler_step_every_batch is False:
                lr_scheduler.step()

            print(f"avg epoch loss: {epoch_loss * batch_size / size:>7f}  [{epoch + 1:>5d}/{epochs:>5d}]")
            print("\n")

        if writer is not None:
            writer.close()

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

    def get_optimizer(self, params, learning_rate=2e-4, betas=(0.9, 0.9), weight_decay=0):
        """Returns Adam optimizer used for training FNAFNet. 

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """

        return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer, T_max=1000, eta_min=1e-6):
        """Return Cosine Annealing Learning Rate Scheduler used for training 
        FNAFNet."""

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
