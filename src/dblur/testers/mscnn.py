from dblur.testers.base_tester import BaseTester
from dblur.data.inference_dataset import InferenceDataset
from dblur.utils.img_utils import save_img, display_img
from dblur.models.mscnn import MSCNN
from dblur.losses.mscnn import MSCNNLoss
import torch
from torchvision.io import read_image
import os
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class MSCNNTester(BaseTester):
    """MSCNN tester for image deblurring. 

    MSCNNTester is subclassed from BaseTester. The class contains methods to 
    test a model used for deblurring, deblur multiple/single images given a 
    pretrained model, get test dataloader, get model and get loss function."""

    def deblur_imgs(self, model, model_path, blur_img_dir, sharp_img_dir, batch_size=2, is_checkpoint=False,
                    window_slicing=False, window_size=256, overlap_size=0):
        """See base class"""

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
        upscale_factor = model.upscale_factor

        blur_img_paths = os.listdir(blur_img_dir)
        sharp_img_paths = [os.path.join(sharp_img_dir, blur_img) for blur_img in blur_img_paths]

        count = 0
        with torch.no_grad():
            for X in dataloader:
                if window_slicing:
                    pred = self.sliding_window_deblur(model, X, window_size, overlap_size)
                else:
                    B1 = X + 0.5
                    height, width = B1.shape[2], B1.shape[3]
                    resize_transform1 = Resize(
                        (int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                    resize_transform2 = Resize(
                        (int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                    B2 = resize_transform1(B1)
                    B3 = resize_transform2(B1)
                    X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                    X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)
                    pred = model(X)

                pred = pred[0]
                denormalized_pred = pred + 0.5
                denormalized_pred = torch.clamp(denormalized_pred, 0, 1)

                for i in range(denormalized_pred.shape[0]):
                    denormalized_pred_img = denormalized_pred[i]
                    torch.clamp(denormalized_pred_img, 0, 1)
                    save_img(denormalized_pred_img, sharp_img_paths[count])
                    count += 1

        print("Successfully saved deblurred images in: ", sharp_img_dir)

    def deblur_single_img(self, model, model_path, blur_img_path, sharp_img_path, is_checkpoint=False,
                          window_slicing=False, window_size=256, overlap_size=0):
        """See base class"""

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

        with torch.no_grad():
            if window_slicing:
                blur_img = blur_img.unsqueeze(0)
                pred = self.sliding_window_deblur(model, blur_img, window_size, overlap_size)
                pred = pred.squeeze(0)
            else:
                B1 = blur_img
                height, width = B1.shape[1], B1.shape[2]
                upscale_factor = model.upscale_factor
                resize_transform1 = Resize((int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                resize_transform2 = Resize((int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                B2 = resize_transform1(B1)
                B3 = resize_transform2(B1)
                blur_img = [B1.float().unsqueeze(0), B2.float().unsqueeze(0), B3.float().unsqueeze(0)]
                pred = model(blur_img)
                pred = pred[0].squeeze(0)

            denormalized_pred = pred + 0.5
            denormalized_pred = torch.clamp(denormalized_pred, 0, 1)

        save_img(denormalized_pred, sharp_img_path)
        display_img(denormalized_pred)

    def test(self, model, model_path, test_dataloader, loss_func, is_checkpoint=False, window_slicing=False,
             window_size=256, overlap_size=0):
        """See base class"""

        if is_checkpoint:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path))

        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        running_loss = 0

        upscale_factor = model.upscale_factor
        psnr = PeakSignalNoiseRatio().to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_score = 0
        ssim_score = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                if window_slicing:
                    pred = self.sliding_window_deblur(model, X, window_size, overlap_size)
                    y = y.to(device)
                else:
                    B1 = X + 0.5
                    S1 = y + 0.5
                    height, width = S1.shape[2], S1.shape[3]
                    resize_transform1 = Resize(
                        (int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                    resize_transform2 = Resize(
                        (int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
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

        if not window_slicing:
            print(f"\ntest loss: {running_loss / len(test_dataloader):>7f}")

        print(f"psnr score: {psnr_score / len(test_dataloader):>7f}")
        print(f"ssim score: {ssim_score / len(test_dataloader):>7f}")

    def sliding_window_deblur(self, model, blur_img, window_size=256, overlap_size=0):
        """See base class"""

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        height, width = blur_img.shape[2], blur_img.shape[3]

        stride = window_size - overlap_size
        pred = torch.zeros(blur_img.shape).to(device)
        num_of_repetitions = torch.ones(blur_img.shape).to(device)
        upscale_factor = model.upscale_factor

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

                window = blur_img[:, :, i:x, j:y]
                B1 = window + 0.5
                h, w = B1.shape[2], B1.shape[3]
                resize_transform1 = Resize((int(h / upscale_factor ** 1), int(w / upscale_factor ** 1)))
                resize_transform2 = Resize((int(h / upscale_factor ** 2), int(w / upscale_factor ** 2)))
                B2 = resize_transform1(B1)
                B3 = resize_transform2(B1)
                X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)

                window_pred = model(X)
                pred[:, :, i:x, j:y] = pred[:, :, i:x, j:y] + window_pred[0]

                if i != 0 and j != 0:
                    num_of_repetitions[:, :, i:i + overlap_size, j:j + y] += 1
                    num_of_repetitions[:, :, i + overlap_size:i + x, j:j + overlap_size] += 1
                elif i == 0 and j != 0:
                    num_of_repetitions[:, :, i:i + x, j:j + overlap_size] += 1
                elif i != 0 and j == 0:
                    num_of_repetitions[:, :, i:i + overlap_size, j:j + y] += 1

        pred = torch.div(pred, num_of_repetitions)

        return pred

    def get_model(self, upscale_factor=2, res_blocks_per_layer=19, channels_dim=64, kernel_size=5):
        """Returns an instance of MSCNN model given the arguments below.

        All arguments have default values set as in the paper. For more details
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

    def get_loss(self):
        """Returns MSCNN model loss."""

        return MSCNNLoss()
