from piq import brisque
import torch
import torch.nn as nn
import numpy as np
import cv2


def calculate_mse(img1, img2):
    """Calculates mse between two images.

    Args:
        img1: Image 1 represented as an instance of torch.Tensor
        img2: Image 2 represented as an instance of torch.Tensor
    """

    return nn.MSELoss(img1, img2)


def calculate_psnr(img1, img2, crop_border=0):
    """Calculates psnr between two images.
    
    Args:
        img1: Image 1 represented as an instance of torch.Tensor
        img2: Image 2 represented as an instance of torch.Tensor
        crop_border: pixels cropped in the border of the images before computing
            psnr.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border=0):
    """Calculates ssim between two images.
    
    Args:
        img1: Image 1 represented as an instance of torch.Tensor
        img2: Image 2 represented as an instance of torch.Tensor
        crop_border: pixels cropped in the border of the images before computing
            ssim.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if img1.dtype is not np.uint8:
        img1 = (img1 * 255.0).round().astype(np.uint8)  # float32 to uint8
    if img2.dtype is not np.uint8:
        img2 = (img2 * 255.0).round().astype(np.uint8)  # float32 to uint8

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))

    return np.array(ssims).mean()


def calculate_brisque(img):
    """Calculates Brisque Index of an image.
    
    For more details regarding Brisque, refer to the paer "No-Reference Image 
    Quality Assessment in the Spatial Domain".
    
    Args:
        img: image represented as an instance of torch.Tensor 
    """

    brisque_index: torch.Tensor = brisque(img.unsqueeze(0), data_range=1.)

    return brisque_index
