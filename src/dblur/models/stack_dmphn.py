import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride,
                      padding=padding)
        )

    def forward(self, x):
        return self.block(x) + x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels=32),
            ConvBlock(channels=32)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            ConvBlock(channels=64),
            ConvBlock(channels=64)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            ConvBlock(channels=128),
            ConvBlock(channels=128)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            ConvBlock(channels=128),
            ConvBlock(channels=128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            ConvBlock(channels=64),
            ConvBlock(channels=64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        )

        self.layer3 = nn.Sequential(
            ConvBlock(channels=32),
            ConvBlock(channels=32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class DMPHN(nn.Module):
    def __init__(self, encoder=Encoder(), decoder=Decoder(), num_layers=4):
        super(DMPHN, self).__init__()
        self.num_layers = num_layers
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            self.encoders.append(encoder)
            self.decoders.append(decoder)

    def get_list_of_blur_patches(self, x):
        blur_imgs = [[x]]
        for i in range(self.num_layers - 1):
            height, width = blur_imgs[i][0].shape[2], blur_imgs[i][0].shape[3]
            blur_imgs_in_layer = []
            for j in range(len(blur_imgs[i])):
                if i % 2 == 1:
                    blur_imgs_in_layer.append(blur_imgs[i][j][:, :, 0:height // 2, :])
                    blur_imgs_in_layer.append(blur_imgs[i][j][:, :, (height + 1) // 2:height, :])
                else:
                    blur_imgs_in_layer.append(blur_imgs[i][j][:, :, :, 0:width // 2])
                    blur_imgs_in_layer.append(blur_imgs[i][j][:, :, :, (width + 1) // 2:width])

            blur_imgs.append(blur_imgs_in_layer)

        return blur_imgs

    def forward(self, x):
        blur_imgs = self.get_list_of_blur_patches(x)
        intermediate_outs = [None] * self.num_layers
        final_outs = [None] * self.num_layers

        for i in range(self.num_layers - 1, -1, -1):
            intermediate_out_i = []
            for j in range(len(blur_imgs[i])):
                if i == self.num_layers - 1:
                    intermediate_out_i.append(self.encoders[i](blur_imgs[i][j]))
                else:
                    intermediate_out_i.append(self.encoders[i](blur_imgs[i][j] + final_outs[i + 1][j]))

            intermediate_out_i_cat = []
            if i == 0:
                intermediate_out_i_cat = [intermediate_out_i[0]]
            else:
                for j in range(int(len(blur_imgs[i]) / 2)):
                    if i % 2 == 1:
                        intermediate_out_i_cat.append(
                            torch.cat([intermediate_out_i[2 * j], intermediate_out_i[2 * j + 1]], dim=3))
                    else:
                        intermediate_out_i_cat.append(
                            torch.cat([intermediate_out_i[2 * j], intermediate_out_i[2 * j + 1]], dim=2))

            intermediate_outs[i] = intermediate_out_i_cat

            final_outs_i = []
            for j in range(len(intermediate_outs[i])):
                if i == self.num_layers - 1:
                    final_outs_i.append(self.decoders[i](intermediate_outs[i][j]))
                elif i == 0:
                    final_outs_i.append(self.decoders[i](intermediate_outs[i][j] + intermediate_outs[i + 1][j]))
                else:
                    if i % 2 == 1:
                        previous_intermediate_out = torch.cat(
                            [intermediate_outs[i + 1][2 * j], intermediate_outs[i + 1][2 * j + 1]], dim=3)
                        final_outs_i.append(self.decoders[i](intermediate_outs[i][j] + previous_intermediate_out))
                    else:
                        previous_intermediate_out = torch.cat(
                            [intermediate_outs[i + 1][2 * j], intermediate_outs[i + 1][2 * j + 1]], dim=2)
                        final_outs_i.append(self.decoders[i](intermediate_outs[i][j] + previous_intermediate_out))

            final_outs[i] = final_outs_i

        return final_outs[0]


class StackDMPHN(nn.Module):
    """
    StackDMPHN model for image deblurring. 
    
    Details regarding the model architecture can be found in the paper "Deep 
    Stacked Hierarchical Multi-patch Network for Image Deblurring".  
    """

    def __init__(self, num_of_stacks=4, encoder=Encoder(), decoder=Decoder(), num_layers=4):
        """Constructor for StackDMPHN. Default values have set for each argument as per the paper.
        
        Args:
            num_of_stacks: number of stacks in each DMPHN module.
            encoder: encoder used in each layer of a DMPHN module.
            decoder: decoded used in each layer of a DMPHN module
            num_layers: number of DMPHN modules.
        """

        super(StackDMPHN, self).__init__()
        self.DMPHN_list = nn.ModuleList()
        self.num_of_stacks = num_of_stacks
        for i in range(num_of_stacks):
            self.DMPHN_list.append(DMPHN(encoder=encoder, decoder=decoder, num_layers=num_layers))

    def forward(self, x):
        out = []
        for i in range(self.num_of_stacks):
            if i == 0:
                out.append(self.DMPHN_list[i](x)[0])
            else:
                out.append(self.DMPHN_list[i](x)[0])

        return out
