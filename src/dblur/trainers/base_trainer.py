import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from dblur.data.dataset import ImageDataset
from dblur.data.augment import RandomCrop


class BaseTrainer:
    """BaseTrainer class
    
    Base trainer class which is used as the parent class by all trainer 
    classes of specific models. The class contains methods to train and validate 
    a model used for deblurring. It also contains methods that return the train 
    and validation dataloaders which are essentially instances of 
    torch.utils.dataDataloader. The trainer class for a specific model will be 
    inherited from BaseTrainer but may overwrite these methods if required. 
    The BaseTrainer class serves only as a skeleton for the trainer class of a 
    specific model. Hence when working with a specific model, an instance of the
    trainer class for that model must be created. (i.e for Restormer, you would
    create an instance of RestormerTrainer for training, validating etc.)"""

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

        """Trains model and performs validation on model at a given frequency given all the arguments below.

        Args:
            model: instance of model to be trained (e.g Restormer, MSCNN, etc.)
            train_dataloader: instance of torch.utils.dataDataloader which 
                iterates through the training data. Can be obtained with the 
                method get_train_dataloader.
            val_dataloader: instance of torch.utils.dataDataloader which 
                iterates through the validation data. Can be obtained with the 
                method get_val_dataloader.
            optimizer: optimizer used for training. (e.g Adam)
            loss_func: loss function used for training.
            lr_scheduler: learning rate scheduler.
            scheduler_step_every_batch: specifies lr_scheduler should take a 
                step every batch or every epoch.
            epochs: number of epochs for training.
            checkpoint_save_name: name of checkpoint (includes model, optimizer, 
                lr_scheduler) to be saved at a given frequency.
            save_checkpoint_freq: frequency at which checkpoint is saved.
            val_freq: frequency at which validation is performed.
            write_logs: If True, logs (includes loss, psnr, ssim) will be 
                plotted on Tensorboard.
            logs_folder: folder in which to write logs.
        """

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
                loss.backward()
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

                if seq_count % val_freq == 0 and val_dataloader is not None:
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

    def validate(self, model, val_dataloader, loss_func, device, batch_count, writer=None):
        """Performs validation on model given all the arguments listed below.

        Args:
            model: instance of the model (e.g Restormer, MSCNN, etc.)
            val_dataloader: instance of torch.utils.data.Dataloader which 
                iterates through the validation data. Can be obtained with the 
                method get_val_dataloader.
            loss_func: loss function used for validation.
            device: cpu or cuda.
            batch_count: indicates the number of times validation has already 
                been performed in a given run.
            writer: instance of SummaryWriter.

        """

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

    def get_train_dataloader(self, dataset_path, transform=RandomCrop(256), batch_size=8):
        """Returns an instance of torch.utils.data.DataLoader used for training given the arguments below.

        Args:
            dataset_path: Path in which dataset for image deblurring is stored.
            transform: Transforms (i.e data augmentations) to apply to dataset
                for training. 
            batch_size: Traning batch size. 
        """

        train_dataset = ImageDataset(dataset_path, transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader

    def get_val_dataloader(self, dataset_path, transform=RandomCrop(256), batch_size=8):
        """Returns an instance of torch.utils.data.DataLoader used for validation given the arguments below.

        Args:
            dataset_path: Path in which dataset for image deblurring is stored.
            transform: Transforms (i.e data augmentations) to apply to dataset
                for validation.
            batch_size: Validation batch size. 
        """

        val_dataset = ImageDataset(dataset_path, transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return val_dataloader
