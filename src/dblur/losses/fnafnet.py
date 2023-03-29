from dblur.losses.charbonnier_loss import CharbonnierLoss
from dblur.losses.frequency_reconstruction_loss import FrequencyReconstructionLoss
import torch.nn as nn


class FNAFNetLoss(nn.Module):
    """FNAFNet Loss

    FAFNet loss for image deblurring used for training NAFNet. It is a 
    combination of the Charbonnier Loss and the Frequency Reconstruction Loss.
    Details regarding the loss function can be found in the paper "Intriguing 
    Findings of Frequency Selection for Image Deblurring".
    """

    def __init__(self, alpha=0.01):
        super(FNAFNetLoss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss()
        self.freq_reconstruction_loss = FrequencyReconstructionLoss()
        self.alpha = alpha

    def forward(self, output, target):
        return self.charbonnier_loss(output, target) + self.alpha * self.freq_reconstruction_loss(output, target)
