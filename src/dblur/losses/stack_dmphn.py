import torch.nn as nn


class StackDMPHNLoss(nn.Module):
    """StackDMPHN Loss.

    StackDMPHN Loss for training StackDMPHN. For more details, refer to paper
    "Deep Stacked Hierarchical Multi-patch Network for Image Deblurring"."""

    def __init__(self):
        super(StackDMPHNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        loss = 0
        for i in range(len(output)):
            loss += self.mse_loss(output[i], target)

        loss = loss / 2
        return loss
