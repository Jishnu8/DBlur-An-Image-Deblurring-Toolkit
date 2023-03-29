import torch
import torch.nn as nn


class MSCNNLoss(nn.Module):
    """MSCNN Model Loss.

    MSCNN Loss used for training an MSCNN model. More details can be found in
    the paper "Deep Multi-scale Convolutional Neural Network for Dynamic Scene 
    Deblurring". 
    """

    def __init__(self):
        super(MSCNNLoss, self).__init__()
        self.no_of_layers = 3
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        L1, L2, L3 = output[0], output[1], output[2]
        S1, S2, S3 = target[0], target[1], target[2]
        loss = (self.mse_loss(L1, S1) + self.mse_loss(L2, S2) + self.mse_loss(L3, S3)) / (2 * self.no_of_layers)

        return loss


class MSCNNDiscriminatorLoss(nn.Module):
    """MSCNN Discriminator Loss.

    MSCNN Discriminator Loss used for training an MSCNN model. More details
    can be found in the paper "Deep Multi-scale Convolutional Neural Network for
    Dynamic Scene Deblurring". 
    """

    def __init__(self):
        super(MSCNNDiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, output_pred, target_pred):
        batch_size = output_pred.shape[0]
        ones = torch.ones((batch_size, 1), device=self.device)
        zeros = torch.zeros((batch_size, 1), device=self.device)

        target_pred_loss = self.bce_loss(target_pred, ones)
        output_pred_loss = self.bce_loss(output_pred, zeros)
        loss = target_pred_loss + output_pred_loss

        return loss
