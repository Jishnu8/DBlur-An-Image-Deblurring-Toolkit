from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop, RandomHorizontalFlip, RotationTransform, RandomSwapColorChannels, \
    RandomSaturation, AddRandomNoise, RandomVerticalFlip
from dblur.trainers.base_trainer import BaseTrainer
from dblur.models.mscnn import MSCNN, MSCNNDiscriminator
from dblur.losses.mscnn import MSCNNLoss, MSCNNDiscriminatorLoss
from dblur.schedulers.mscnn import MSCNNScheduler
import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class MSCNNTrainer(BaseTrainer):
    """MSCNN trainer for image deblurring. 
    
    MSCNNTrainer is subclassed from BaseTrainer. It contains methods to train,
    validate, get model, get discriminator get model loss function, get 
    discriminator loss function, get optimizer, get learning rate scheduler, 
    get train dataloader and get validation dataloader. The default
    arguments in each method has been given a value as specified in the paper 
    "Deep Multi-scale Convolutional Neural Network for Dynamic Scene 
    Deblurring". Fore more details regarding each argument in each method, refer 
    to the paper.
    """

    def train(
            self,
            model=None,
            discriminator=None,
            train_dataloader=None,
            val_dataloader=None,
            model_optimizer=None,
            discriminator_optimizer=None,
            model_loss_func=None,
            discriminator_loss_func=None,
            lr_scheduler=None,
            scheduler_step_every_batch=False,
            epochs=2000,
            checkpoint_save_name=None,
            save_checkpoint_freq=300,
            val_freq=100,
            write_logs=False,
            logs_folder='runs',
            weight_constant=1e-5
    ):

        """See base class"""

        if train_dataloader is None:
            raise Exception("train_dataloader was not provided.")
        if checkpoint_save_name is None:
            raise Exception("Path to directory for saving model was not provided.")
        if model is None and model_optimizer is None:
            print("Loading default model specified in get_model() as no model was provided")
            model = self.get_model()
            print(
                "Loading default model optimizer specified in get_model_optimizer() as no model optimizer was provided")
            model_optimizer = self.get_model_optimizer(model.parameters())
        if discriminator is None and discriminator_optimizer is None:
            print("Loading default discriminator specified in get_discriminator() as no discriminator was provided")
            model = self.get_discriminator()
            print("Loading default discriminator optimizer specified in get_discriminator_optimizer() as no "
                  "discriminator optimizer was provided")
            discriminator_optimizer = self.get_optimizer()
        if model_loss_func is None:
            print("Loading default model loss function specified in get_model_loss() as no model loss was provided")
            model_loss_func = self.get_model_loss()
        if discriminator_loss_func is None:
            print("Loading default discriminator loss function specified in get_discriminator_loss() as no "
                  "discriminator loss was provided")
            discriminator_loss_func = self.get_discriminator_loss()
        if val_dataloader is None:
            print("Warning. val_dataloader was not provided.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if write_logs:
            writer = SummaryWriter(logs_folder)
        else:
            writer = None

        model = model.to(device)
        discriminator = discriminator.to(device)

        if os.path.exists(checkpoint_save_name):
            checkpoint = torch.load(checkpoint_save_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print("Existing model loaded")

        model.train()

        upscale_factor = model.upscale_factor
        seq_count = 0
        batch_size = train_dataloader.batch_size
        size = len(train_dataloader.dataset)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            epoch_loss = 0
            for g in model_optimizer.param_groups:
                print("Learning rate: ", g['lr'])
            seq_batch_count = 0
            for train_batch, (X, y) in enumerate(train_dataloader):
                B1 = X + 0.5
                S1 = y + 0.5
                height, width = S1.shape[2], S1.shape[3]
                resize_transform1 = Resize((int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                resize_transform2 = Resize((int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                S2 = resize_transform1(S1)
                B2 = resize_transform1(B1)
                S3 = resize_transform2(S1)
                B3 = resize_transform2(B1)
                X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                y = [(S1 - 0.5).float(), (S2 - 0.5).float(), (S3 - 0.5).float()]

                X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)
                y[0], y[1], y[2] = y[0].to(device), y[1].to(device), y[2].to(device)

                for p in discriminator.parameters():
                    p.requires_grad = True

                pred = model(X)
                dis_output_pred = discriminator(pred[0])
                dis_target_pred = discriminator(y[0])
                discriminator_loss = discriminator_loss_func(dis_output_pred, dis_target_pred)
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimizer.step()

                for p in discriminator.parameters():
                    p.requires_grad = False

                dis_target_pred = discriminator(y[0])
                dis_output_pred = discriminator(pred[0])
                loss = model_loss_func(pred, y) + weight_constant * discriminator_loss_func(dis_output_pred,
                                                                                            dis_target_pred)
                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()

                seq_count += 1
                seq_batch_count += 1
                current = (train_batch + 1) * batch_size
                if current > size:
                    current = size

                print(f"model loss: {loss.detach().data:>7f}  [{current:>5d}/{size:>5d}]")

                epoch_loss += loss.item()
                if write_logs:
                    writer.add_scalar("train_loss", float(loss.detach().data), seq_count)

                if seq_count % save_checkpoint_freq == 0:
                    if lr_scheduler is not None:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'discriminator_state_dict': discriminator.state_dict(),
                            'model_optimizer_state_dict': model_optimizer.state_dict(),
                            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict()
                        }, checkpoint_save_name)
                    else:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'discriminator_state_dict': discriminator.state_dict(),
                            'model_optimizer_state_dict': model_optimizer.state_dict(),
                            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict()
                        }, checkpoint_save_name)

                if seq_count % val_freq == 0:
                    self.validate(model, val_dataloader, model_loss_func, device, seq_count // val_freq, writer)
                    model.train()

                if lr_scheduler is not None and scheduler_step_every_batch is True:
                    lr_scheduler.step()

            if lr_scheduler is not None and scheduler_step_every_batch is False:
                lr_scheduler.step()

            print(f"avg epoch loss: {epoch_loss * batch_size / size:>7f}  [{epoch + 1:>5d}/{epochs:>5d}]")
            print("\n")

        if writer is not None:
            writer.close()

    def validate(self, model, val_dataloader, loss_func, device, batch_count, writer=None):
        """See base class"""

        model.eval()
        running_loss = 0

        psnr = PeakSignalNoiseRatio().to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_score = 0
        ssim_score = 0
        upscale_factor = model.upscale_factor

        with torch.no_grad():
            for X, y in val_dataloader:
                B1 = X + 0.5
                S1 = y + 0.5
                height, width = S1.shape[2], S1.shape[3]
                resize_transform1 = Resize((int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                resize_transform2 = Resize((int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                S2 = resize_transform1(S1)
                B2 = resize_transform1(B1)
                S3 = resize_transform2(S1)
                B3 = resize_transform2(B1)
                X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                y = [(S1 - 0.5).float(), (S2 - 0.5).float(), (S3 - 0.5).float()]

                X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)
                y[0], y[1], y[2] = y[0].to(device), y[1].to(device), y[2].to(device)

                pred = model(X)
                loss = loss_func(pred, y)
                running_loss += loss
                pred = pred[0]
                y = y[0]

                denormalized_pred = pred + 0.5
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

    def get_model(self, upscale_factor=2, res_blocks_per_layer=19, channels_dim=64, kernel_size=5):
        """Returns an instance of MSCNN model given the arguments below. All 
        arguments have default values set as in the paper. For more details 
        regarding the arguments below, refer to the paper "Deep Multi-scale 
        Convolutional Neural Network for Dynamic Scene Deblurring".

        Args:
            upscale_factor: scaling factor of input blurry image at each layer 
                of MSCNN. By default, upscale_factor = 2. Hence layer Lk in the 
                MSCNN model takes the blurry image as input donwsampled to 1/2^k
                scale.
            res_blocks_per_layer: number of ResBlocks in each layer.
            channels_dim: number of channels the first convolution layer 
                projects the image into.
            kernel_size: kernel size of the convolution layer present in each
                ResBlock.         
        """

        return MSCNN(upscale_factor=upscale_factor, res_blocks_per_layer=res_blocks_per_layer,
                     channels_dim=channels_dim, kernel_size=kernel_size)

    def get_discriminator(self, negative_slope=0.01):
        """Returns an instance of MSCNN discriminator used for training.

        Args:
            negative_slope: slope of ReLU activation used in model architecture
            of MSCNNDiscriminator. 
        """

        return MSCNNDiscriminator(negative_slope=negative_slope)

    def get_model_loss(self):
        """Returns MSCNN model loss."""
        return MSCNNLoss()

    def get_discriminator_loss(self):
        """Returns MSCNN discriminator loss."""
        return MSCNNDiscriminatorLoss()

    def get_model_optimizer(self, params, learning_rate=5e-5, betas=(0.9, 0.999), weight_decay=0):
        """Returns Adam optimizer used for training MSCNN specified by the arguments below:

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """

        return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_discriminator_optimizer(self, params, learning_rate=5e-5, betas=(0.9, 0.999), weight_decay=0):
        """Returns Adam optimizer used for MSCNN discriminator which is used in training.

        Args:
            params: parameters of the model to be trained. (typically 
                model.parameters())
            learning_rate: learning rate of optimizer
            betas: beta values for Adam
            weight_decay: weight decay (L2 penalty)
        """
        return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer, steps_for_change=3e5, decrease_factor=0.1):
        """Learning rate scheduler used for training MSCNN referred to in the paper."""

        return MSCNNScheduler(optimizer, steps_for_change=steps_for_change, decrease_factor=decrease_factor)

    def get_train_dataloader(self, dataset_path, transform="default", batch_size=2):
        """See base class"""

        if transform == "default":
            train_composed = transforms.Compose([RandomCrop(256),
                                                 RandomHorizontalFlip(),
                                                 RandomVerticalFlip(),
                                                 RotationTransform(angles=[0, 90, 270]),
                                                 RandomSwapColorChannels(),
                                                 RandomSaturation(),
                                                 AddRandomNoise(mu=0, sigma=2 / 255)
                                                 ])
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
            val_composed = transforms.Compose([RandomCrop(256)])
        else:
            val_composed = transform

        val_dataset = ImageDataset(dataset_path, val_composed)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return val_dataloader
