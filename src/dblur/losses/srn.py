import torch.nn as nn
from torchvision.transforms import Resize


class SRNLoss(nn.Module):
    """
    SRN loss for image deblurring. 
    
    Details regarding the loss function can be found in the paper 
    "Scale-recurrent Network for Deep Image Deblurring".  
    """

    def __init__(self):
        super(SRNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        loss = 0
        num_of_layers = len(output)
        resize_imgs = []
        height, width = target.shape[2], target.shape[3]
        idx = 0
        for i in range(num_of_layers - 1, 0, -1):
            resize_transform = Resize((int(height / 2 ** i), int(width / 2 ** i)))
            resize_imgs.append(resize_transform(target))
            idx += 1

        resize_imgs.append(target)

        for i in range(len(output)):
            loss += self.mse_loss(output[i], resize_imgs[i])

        return loss
