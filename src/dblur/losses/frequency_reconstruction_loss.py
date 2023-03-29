import torch
import torch.nn as nn


class FrequencyReconstructionLoss(nn.Module):
    """Frequency Reconstruction Loss"""

    def forward(self, output, target):
        output_fft = torch.fft.rfft2(output)
        target_fft = torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(output_fft - target_fft))

        return loss
