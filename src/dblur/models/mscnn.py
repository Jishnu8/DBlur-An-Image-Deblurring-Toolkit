import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, input_dim=64, kernel_size=5):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(self.input_dim, self.input_dim, kernel_size=self.kernel_size, stride=1, padding="same")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.input_dim, self.input_dim, kernel_size=self.kernel_size, stride=1, padding="same")

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class UpConvBlock(nn.Module):
    def __init__(self, upscale_factor=2, kernel_size=5):
        super(UpConvBlock, self).__init__()
        self.conv = nn.Conv2d(3, 3 * upscale_factor ** 2, kernel_size=kernel_size, stride=1, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class MSCNNLayer(nn.Module):
    def __init__(self, no_of_res_blocks=19, input_dim=3, channels_dim=64, kernel_size=5):
        super(MSCNNLayer, self).__init__()
        self.no_of_res_blocks = no_of_res_blocks
        self.channels_dim = channels_dim
        self.kernel_size = kernel_size

        self.input_conv = nn.Conv2d(input_dim, self.channels_dim, self.kernel_size, stride=1, padding="same")
        self.output_conv = nn.Conv2d(self.channels_dim, 3, self.kernel_size, stride=1, padding="same")
        self.list_of_res_blocks = []
        for i in range(self.no_of_res_blocks):
            self.list_of_res_blocks.append(ResBlock(input_dim=self.channels_dim, kernel_size=self.kernel_size))

        self.layers = nn.Sequential(*self.list_of_res_blocks)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.layers(x)
        x = self.output_conv(x)

        return x


class MSCNN(nn.Module):
    """
    MSCNN model for image deblurring. 
    
    Details regarding the model architecture can be found in the paper "Deep 
    Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring".  
    """

    def __init__(self, upscale_factor=2, res_blocks_per_layer=19, channels_dim=64, kernel_size=5):
        """
        Constructor of MSCNN. Default values have set for each argument as the paper.

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

        super(MSCNN, self).__init__()
        self.no_of_layers = 3
        self.upscale_factor = upscale_factor

        self.mscnn_layer1 = MSCNNLayer(no_of_res_blocks=res_blocks_per_layer, input_dim=3, channels_dim=channels_dim,
                                       kernel_size=kernel_size)
        self.up_conv1 = UpConvBlock(upscale_factor=upscale_factor, kernel_size=kernel_size)
        self.mscnn_layer2 = MSCNNLayer(no_of_res_blocks=res_blocks_per_layer, input_dim=6, channels_dim=channels_dim,
                                       kernel_size=kernel_size)
        self.up_conv2 = UpConvBlock(upscale_factor=upscale_factor, kernel_size=kernel_size)
        self.mscnn_layer3 = MSCNNLayer(no_of_res_blocks=res_blocks_per_layer, input_dim=6, channels_dim=channels_dim,
                                       kernel_size=kernel_size)

    def forward(self, x):
        B1, B2, B3 = x[0], x[1], x[2]
        L3 = self.mscnn_layer1(B3)
        L2 = self.mscnn_layer2(torch.cat((self.up_conv1(L3), B2), axis=1))
        L1 = self.mscnn_layer3(torch.cat((self.up_conv2(L2), B1), axis=1))

        return [L1, L2, L3]


class MSCNNDiscriminatorBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, stride=1, padding=2, bias=False, negative_slope=0.01):
        super(MSCNNDiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(negative_slope, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MSCNNDiscriminator(nn.Module):
    """
    MSCNN discriminator used for training a MSCNN model. 
    
    Details regarding the model architecture can be found in the paper "Deep 
    Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring".  
    """

    def __init__(self, negative_slope=0.01):
        super(MSCNNDiscriminator, self).__init__()
        self.model = nn.Sequential(
            MSCNNDiscriminatorBlock(3, 32, stride=1, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(32, 32, stride=2, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(32, 64, stride=1, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(64, 64, stride=2, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(64, 128, stride=1, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(128, 128, stride=4, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(128, 256, stride=1, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(256, 256, stride=4, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(256, 512, stride=1, negative_slope=negative_slope),
            MSCNNDiscriminatorBlock(512, 512, kernel_size=4, stride=4, padding=0, negative_slope=negative_slope)
        )

        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = torch.sigmoid(x)
        return out
