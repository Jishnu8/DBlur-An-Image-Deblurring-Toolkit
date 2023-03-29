import torch
import torch.nn as nn
from torchvision.transforms import Resize


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding)
        )

    def forward(self, x):
        return self.conv_block(x) + x


class InBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1, padding=2):
        super(InBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ResBlock(in_channels=channels, out_channels=channels),
            ResBlock(in_channels=channels, out_channels=channels),
            ResBlock(in_channels=channels, out_channels=channels)
        )

    def forward(self, x):
        return self.model(x)


class OutBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1, padding=2):
        super(OutBlock, self).__init__()
        self.model = nn.Sequential(
            ResBlock(in_channels=channels, out_channels=channels),
            ResBlock(in_channels=channels, out_channels=channels),
            ResBlock(in_channels=channels, out_channels=channels),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.model(x)


class EBlock(nn.Module):
    def __init__(self, in_channels):
        super(EBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            ResBlock(in_channels=in_channels * 2, out_channels=in_channels * 2),
            ResBlock(in_channels=in_channels * 2, out_channels=in_channels * 2),
            ResBlock(in_channels=in_channels * 2, out_channels=in_channels * 2)
        )

    def forward(self, x):
        return self.model(x)


class DBlock(nn.Module):
    def __init__(self, in_channels):
        super(DBlock, self).__init__()
        self.model = nn.Sequential(
            ResBlock(in_channels=in_channels, out_channels=in_channels),
            ResBlock(in_channels=in_channels, out_channels=in_channels),
            ResBlock(in_channels=in_channels, out_channels=in_channels),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=4, stride=2,
                               padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class SRNLayer(nn.Module):
    def __init__(self, num_of_e_blocks=2):
        super(SRNLayer, self).__init__()
        encoder = []
        decoder = []
        projected_in_channels = 32
        self.num_of_e_blocks = num_of_e_blocks

        encoder.append(InBlock(projected_in_channels))
        for j in range(num_of_e_blocks):
            encoder.append(EBlock(projected_in_channels * 2 ** j))
        self.encoder = nn.Sequential(*encoder)

        self.conv_lstm = ConvLSTMCell(projected_in_channels * (2 ** num_of_e_blocks),
                                      projected_in_channels * (2 ** num_of_e_blocks), kernel_size=(5, 5), bias=True)

        for j in range(num_of_e_blocks, 0, -1):
            decoder.append(DBlock(projected_in_channels * 2 ** j))
        decoder.append(OutBlock(projected_in_channels))
        self.decoder = nn.Sequential(*decoder)
        self.ini_hidden_state = None

    def forward(self, x, hidden_state=None):
        x = self.encoder(x)
        if hidden_state is None:
            if self.ini_hidden_state is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.ini_hidden_state = (
                    torch.zeros(x.shape[0], 32 * (2 ** self.num_of_e_blocks), x.shape[2], x.shape[3],
                                requires_grad=True).to(device),
                    torch.zeros(x.shape[0], 32 * (2 ** self.num_of_e_blocks), x.shape[2], x.shape[3],
                                requires_grad=True).to(device))

            hidden_state = self.ini_hidden_state

        h_next, c_next = self.conv_lstm(x, hidden_state)
        y = self.decoder(c_next)

        return (h_next, c_next), y


class SRN(nn.Module):
    """
    SRN model for image deblurring. 
    
    Details regarding the model architecture can be found in the paper 
    "Scale-recurrent Network for Deep Image Deblurring".  
    """

    def __init__(self, num_of_layers=3, num_of_e_blocks=2):
        """Constructor of SRN. Default values have set for each argument as per the paper.

        Args:
            num_of_layers: number of layers in SRN model
            num_of_e_blocks: number of encoder/decoder blocks in each layer
        """

        super(SRN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_of_layers = num_of_layers
        self.num_of_e_blocks = num_of_e_blocks

        for i in range(num_of_layers):
            self.layers.append(SRNLayer(num_of_e_blocks=num_of_e_blocks))

    def get_resized_imgs(self, x):
        height, width = x.shape[2], x.shape[3]
        resize_imgs = []
        for i in range(self.num_of_layers - 1, 0, -1):
            resize_transform = Resize((int(height / 2 ** i), int(width / 2 ** i)))
            resize_imgs.append(resize_transform(x))

        resize_imgs.append(x)
        return resize_imgs

    def forward(self, x):
        resize_imgs = self.get_resized_imgs(x)
        outs = []
        for i in range(self.num_of_layers):
            if i == 0:
                h_next, out = self.layers[i](resize_imgs[i])
                outs.append(out)
            else:
                height, width = resize_imgs[i].shape[2], resize_imgs[i].shape[3]
                resize_transform = Resize((int(height), int(width)))
                previous_out = resize_transform(outs[i - 1])
                scaled_h_next_shape = (h_next[0].shape[2] * 2, h_next[0].shape[3] * 2)
                h_next_scaled = (torch.nn.functional.interpolate(h_next[0], size=scaled_h_next_shape),
                                 torch.nn.functional.interpolate(h_next[1], size=scaled_h_next_shape))
                h_next, out = self.layers[i](resize_imgs[i] + previous_out, h_next_scaled)
                outs.append(out)

        return outs
