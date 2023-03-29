import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio


class PSNRLoss(nn.Module):
    """
    PSNR loss for image deblurring used for training NAFNet. 
    
    Details regarding the loss function can be found in the paper "Simple 
    Baselines for Image Restoration".  
    """

    def __init__(self):
        super(PSNRLoss, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.psnr = PeakSignalNoiseRatio(data_range=1).to(device)
        self.mse_loss = nn.MSELoss()
        self.max_pixel_value = 255

    def forward(self, output, target):
        print(self.mse_loss(output, target).item())
        return -self.psnr(output, target)
