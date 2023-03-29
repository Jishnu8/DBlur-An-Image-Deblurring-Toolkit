from dblur.testers.base_tester import BaseTester
from dblur.data.dataset import ImageDataset
from dblur.data.inference_dataset import InferenceDataset
from dblur.utils.img_utils import save_img, display_img
from dblur.models.srn import SRN
from dblur.losses.srn import SRNLoss
import torch
from torchvision.io import read_image
import os
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class SRNTester(BaseTester):
    """SRN tester for image deblurring. 

    SRNTester is subclassed from BaseTester. The class contains methods to 
    test a model used for deblurring, deblur multiple/single images given a 
    pretrained model, get test dataloader, get model and get loss function."""

    def get_test_dataloader(self, dataset_path, transform=None, batch_size=6):
        """See base class."""

        test_dataset = ImageDataset(dataset_path, transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=True)

        return test_dataloader

    def deblur_imgs(self, model, model_path, blur_img_dir, sharp_img_dir, batch_size=2, is_checkpoint=True,
                    window_slicing=False, window_size=256, overlap_size=0):
        """See base class."""

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
                denormalized_pred = pred[len(pred) - 1] + 0.5
                denormalized_pred = torch.clamp(denormalized_pred, 0, 1)

                for i in range(denormalized_pred.shape[0]):
                    denormalized_pred_img = denormalized_pred[i]
                    torch.clamp(denormalized_pred_img, 0, 1)
                    save_img(denormalized_pred_img, sharp_img_paths[count])
                    count += 1

        print("Successfully saved deblurred images in: ", sharp_img_dir)

    def deblur_single_img(self, model, model_path, blur_img_path, sharp_img_path, is_checkpoint=True,
                          window_slicing=False, window_size=256, overlap_size=0):
        """See base class."""

        if is_checkpoint:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path))

        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        blur_img = ((read_image(blur_img_path).type(torch.float32)) / 255).to(device)
        blur_img -= 0.5
        blur_img = blur_img.unsqueeze(0)

        with torch.no_grad():
            if window_slicing:
                pred = self.sliding_window_deblur(model, blur_img, window_size, overlap_size)
            else:
                pred = model(blur_img)
            pred = pred[len(pred) - 1].squeeze(0)

            denormalized_pred = pred + 0.5
            denormalized_pred = torch.clamp(denormalized_pred, 0, 1)

        save_img(denormalized_pred, sharp_img_path)
        display_img(denormalized_pred)

    def test(self, model, model_path, test_dataloader, loss_func, is_checkpoint=False, window_slicing=False,
             window_size=256, overlap_size=0):
        """See base class."""

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

                denormalized_pred = pred[len(pred) - 1] + 0.5
                denormalized_y = y + 0.5
                denormalized_pred = torch.clamp(denormalized_pred, 0, 1)
                denormalized_y = torch.clamp(denormalized_y, 0, 1)

                psnr_score += psnr(denormalized_pred, denormalized_y)
                ssim_score += ssim(denormalized_pred, denormalized_y)

        print(f"\ntest loss: {running_loss / len(test_dataloader):>7f}")
        print(f"psnr score: {psnr_score / len(test_dataloader):>7f}")
        print(f"ssim score: {ssim_score / len(test_dataloader):>7f}")

    def sliding_window_deblur(self, model, blur_img, window_size=256, overlap_size=0):
        """See base class."""

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        height, width = blur_img.shape[2], blur_img.shape[3]

        stride = window_size - overlap_size
        num_of_layers = model.num_of_layers
        pred = [torch.zeros(blur_img.shape).to(device) for i in range(num_of_layers)]  # change here
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

                for k in range(num_of_layers):
                    pred[k][:, :, i:x, j:y] = pred[k][:, :, i:x, j:y] + window_pred[k]

                if i != 0 and j != 0:
                    num_of_repetitions[:, :, i:i + overlap_size, j:j + y] += 1
                    num_of_repetitions[:, :, i + overlap_size:i + x, j:j + overlap_size] += 1
                elif i == 0 and j != 0:
                    num_of_repetitions[:, :, i:i + x, j:j + overlap_size] += 1
                elif i != 0 and j == 0:
                    num_of_repetitions[:, :, i:i + overlap_size, j:j + y] += 1

        for i in range(num_of_layers):
            pred[i] = torch.div(pred[i], num_of_repetitions)

        return pred

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
