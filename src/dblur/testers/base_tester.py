import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from dblur.utils.dataset_utils import get_blur_img_paths
from dblur.utils.img_utils import save_img, display_img
from dblur.data.inference_dataset import InferenceDataset
from dblur.data.dataset import ImageDataset


class BaseTester:
    """BaseTester class
    
    Base tester class which is used as the parent class by all tester classes of 
    specific models. The class contains methods to return a test dataloader 
    which is an instance of torch.utils.data.DataLoader, test a model used for 
    deblurring and deblur multiple/single images given a pretrained model. The 
    tester class for a specific model will be inherited from BaseTester but may 
    overwrite these methods if required. 
    The BaseTester class serves only as a skeleton for the tester class of a 
    specific model. Hence when working with a specific model, an instance of the
    tester class for that model must be created. (i.e for Restormer, you would
    create an instance of RestormerTesting for testing, deblurring etc.)"""

    def get_test_dataloader(self, dataset_path, transform=None, batch_size=8):
        """Returns an instance of torch.utils.data.DataLoader used for testing given the arguments below.

        Args:
            dataset_path: Path in which dataset for image deblurring is stored.
            transform: Transforms (i.e data augmentations) to apply to dataset
                for testing.
            batch_size: Testing batch size. 
        """

        test_dataset = ImageDataset(dataset_path, transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return test_dataloader

    def test(self, model, model_path, test_dataloader, loss_func, is_checkpoint=False, window_slicing=False,
             window_size=256, overlap_size=0):
        """Tests model given all the arguments listed below.

        Args:
            model: instance of the model (e.g Restormer, MSCNN, etc.)
            model_path: path of the checkpoint (model, optimizer and 
                lr_scheduler) or just the model. 
            test_dataloader: instance of torch.utils.data.Dataloader which 
                iterates through the testing data. Can be obtained with the 
                method get_test_dataloader.
            loss_func: loss function used for testing.
            is_checkpoint: specifies if path specified in model_path is a 
                checkpoint (model, optimizer and lr_scheduler) or just a model. 
            window_slicing: if True, image is sliced into overlapping windows of
                size window_size and overlap overlap_size. Deblurring is then 
                performed in each window and merged back appropriately. This 
                takes into account the fact that training of deblurring models
                are usually performed on small image patches.
            window_size: size of the window in pixels that the image is sliced 
                into.
            overlap_size: size of overlap between two windows in pixels.  
        """

        if is_checkpoint:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path))

        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        running_loss = 0

        psnr = PeakSignalNoiseRatio().to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_score = 0
        ssim_score = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                if window_slicing:
                    pred = self.sliding_window_deblur(model, X, window_size, overlap_size)
                else:
                    pred = model(X)

                loss = loss_func(pred, y)
                running_loss += loss

                denormalized_pred = pred + 0.5
                denormalized_y = y + 0.5
                denormalized_pred = torch.clamp(denormalized_pred, 0, 1)
                denormalized_y = torch.clamp(denormalized_y, 0, 1)

                psnr_score += psnr(denormalized_pred, denormalized_y)
                ssim_score += ssim(denormalized_pred, denormalized_y)

        print(f"\ntest loss: {running_loss / len(test_dataloader):>7f}")
        print(f"psnr score: {psnr_score / len(test_dataloader):>7f}")
        print(f"ssim score: {ssim_score / len(test_dataloader):>7f}")

    def deblur_imgs(self, model, model_path, blur_img_dir, sharp_img_dir, batch_size=2, is_checkpoint=True,
                    window_slicing=False, window_size=256, overlap_size=0):
        """Deblurs all images in a specified directory given a pretrained model.

        Args:
            model: instance of the model (e.g Restormer, MSCNN, etc.)
            model_path: path of the checkpoint (model, optimizer and 
                lr_scheduler) or just the model. 
            blur_img_dir: directory which contains all the blur images. 
            sharp_img_dir: directory in which all the deblurred images shall be
                stored.
            batch_size: batch size in which to perform deblurring of images. 
            is_checkpoint: specifies if path specified in model_path is a 
                checkpoint (model, optimizer and lr_scheduler) or just a model. 
            window_slicing: if True, image is sliced into overlapping windows of
                size window_size and overlap overlap_size. Deblurring is then 
                performed in each window and merged back appropriately. This 
                takes into account the fact that training of deblurring models
                are usually performed on small image patches.
            window_size: size of the window in pixels that the image is sliced 
                into.
            overlap_size: size of overlap between two windows in pixels.  
        """

        if is_checkpoint:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path))

        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        dataset = InferenceDataset(blur_img_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        blur_img_paths = os.listdir(blur_img_dir)
        sharp_img_paths = [os.path.join(sharp_img_dir, blur_img) for blur_img in blur_img_paths]

        count = 0
        with torch.no_grad():
            for X in dataloader:
                X = X.to(device)
                if window_slicing:
                    pred = self.sliding_window_deblur(model, X, window_size, overlap_size)
                else:
                    pred = model(X)

                denormalized_pred = pred + 0.5
                denormalized_pred = torch.clamp(denormalized_pred, 0, 1)

                for i in range(denormalized_pred.shape[0]):
                    denormalized_pred_img = denormalized_pred[i]
                    torch.clamp(denormalized_pred_img, 0, 1)
                    save_img(denormalized_pred_img, sharp_img_paths[count])
                    count += 1

        print("Successfully saved deblurred images in: ", sharp_img_dir)

    def deblur_single_img(self, model, model_path, blur_img_path, sharp_img_path, is_checkpoint=True,
                          window_slicing=False, window_size=256, overlap_size=0):
        """Deblurs image given a pretrained model, displays and saves the deblurred image.

        Args:
            model: instance of the model (e.g Restormer, MSCNN, etc.)
            model_path: path of the checkpoint (model, optimizer and 
                lr_scheduler) or just the model. 
            blur_img_path: path of the blur image 
            sharp_img_path: path in which the deblurred image shall be stored.
            is_checkpoint: specifies if path specified in model_path is a 
                checkpoint (model, optimizer and lr_scheduler) or just a model. 
            window_slicing: if True, image is sliced into overlapping windows of
                size window_size and overlap overlap_size. Deblurring is then 
                performed in each window and merged back appropriately. This 
                takes into account the fact that training of deblurring models
                are usually performed on small image patches.
            window_size: size of the window in pixels that the image is sliced 
                into.
            overlap_size: size of overlap between two windows in pixels.  
        """

        if is_checkpoint:
            checkpoint = torch.load(model_path)  # , map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path))

        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        blur_img = ((read_image(blur_img_path).type(torch.float32)) / 255).to(device)
        blur_img = torch.clamp(blur_img, 0, 1)
        blur_img -= 0.5
        blur_img = blur_img.unsqueeze(0)

        with torch.no_grad():
            if window_slicing:
                pred = self.sliding_window_deblur(model, blur_img, window_size, overlap_size)
            else:
                pred = model(blur_img)
            pred = pred.squeeze(0)

            denormalized_pred = pred + 0.5
            denormalized_pred = torch.clamp(denormalized_pred, 0, 1)

        save_img(denormalized_pred, sharp_img_path)
        display_img(denormalized_pred)

    def sliding_window_deblur(self, model, blur_img, window_size=256, overlap_size=0):
        """Deblurs image by slicing in into overlapping windows.

        Deblurs image by slicing it into overlapping windows. Each window
        is deblurred using the model and finally they are merged back 
        approriately. This takes into account the fact that training of 
        deblurring models are usually performed on small image patches.

        Args:
            model: instance of the model (e.g Restormer, MSCNN, etc.)
            blur_img: blur image represented by an instance of torch.Tensor
            window_size: size of the window in pixels that the image is sliced 
                into.
            overlap_size: size of overlap between two windows in pixels.  
        """

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        height, width = blur_img.shape[2], blur_img.shape[3]

        i, j = 0, 0
        stride = window_size - overlap_size
        pred = torch.zeros(blur_img.shape).to(device)
        num_of_repetitions = torch.ones(blur_img.shape).to(device)

        for i in range(0, height, stride):
            for j in range(0, width, stride):
                if i + window_size > height and j + window_size <= width:
                    x = height
                    y = j + window_size
                elif i + window_size <= height and j + window_size > width:
                    x = i + window_size
                    y = width
                elif i + window_size <= height and j + window_size <= width:
                    x = i + window_size
                    y = j + window_size
                elif i + window_size > height and j + window_size > width:
                    x = height
                    y = width

                window_pred = model(blur_img[:, :, i:x, j:y])
                pred[:, :, i:x, j:y] = pred[:, :, i:x, j:y] + window_pred

                if i != 0 and j != 0:
                    num_of_repetitions[:, :, i:i + overlap_size, j:j + y] += 1
                    num_of_repetitions[:, :, i + overlap_size:i + x, j:j + overlap_size] += 1
                elif i == 0 and j != 0:
                    num_of_repetitions[:, :, i:i + x, j:j + overlap_size] += 1
                elif i != 0 and j == 0:
                    num_of_repetitions[:, :, i:i + overlap_size, j:j + y] += 1

        pred = torch.div(pred, num_of_repetitions)

        return pred
