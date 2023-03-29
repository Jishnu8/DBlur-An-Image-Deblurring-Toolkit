import torch.nn as nn
import torch


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss """

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        diff = output - target
        loss = torch.sqrt(torch.mean(diff * diff) + self.eps ** 2)

        return loss
