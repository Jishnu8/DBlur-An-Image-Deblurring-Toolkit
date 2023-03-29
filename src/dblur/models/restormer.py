import torch
import torch.nn as nn


class MDTA(nn.Module):
    def __init__(self, channels, num_heads, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.scaling_param = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels * 3, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=channels * 3, out_channels=channels * 3, kernel_size=3, padding=1,
                               groups=channels * 3, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        query, key, value = self.conv2(self.conv1(x)).chunk(3, dim=1)

        query = query.reshape(batch_size, self.num_heads, -1, height * width)
        key = key.reshape(batch_size, self.num_heads, -1, height * width)
        value = value.reshape(batch_size, self.num_heads, -1, height * width)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn = torch.softmax(torch.matmul(query, torch.transpose(key, -2, -1)) * self.scaling_param, dim=-1)
        out = torch.matmul(attn, value)
        out_reshape = out.reshape(batch_size, -1, height, width)
        projected_out = self.conv3(out_reshape)

        return projected_out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor, bias=False):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=hidden_channels * 2, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels * 2, out_channels=hidden_channels * 2, kernel_size=3,
                               padding=1, groups=hidden_channels, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=channels, kernel_size=1, bias=bias)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x1, x2 = self.conv2(self.conv1(x)).chunk(2, dim=1)
        x3 = self.gelu(x2) * x1
        out = self.conv3(x3)
        return out + x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor, bias=False):
        super(TransformerBlock, self).__init__()
        self.layernorm1 = torch.nn.LayerNorm(channels)
        self.mdta = MDTA(channels=channels, num_heads=num_heads, bias=bias)
        self.layernorm2 = torch.nn.LayerNorm(channels)
        self.gdfn = GDFN(channels=channels, expansion_factor=expansion_factor, bias=bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x1 = self.layernorm1(x.reshape(batch_size, channels, height * width).transpose(1, 2))
        mdta_out = self.mdta(x1.transpose(1, 2).reshape(batch_size, channels, height, width))
        x2 = self.layernorm1(mdta_out.reshape(batch_size, channels, height * width).transpose(1, 2))
        gdfn_out = self.gdfn(x2.transpose(1, 2).reshape(batch_size, channels, height, width))

        return gdfn_out


class DownSample(nn.Module):
    def __init__(self, channels, bias=False):
        super(DownSample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, padding=1,
                                    bias=bias)
        self.px_unshuffle = torch.nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.px_unshuffle(x)

        return x


class UpSample(nn.Module):
    def __init__(self, channels, bias=False):
        super(UpSample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=3, padding=1,
                                    bias=bias)
        self.px_shuffle = torch.nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.px_shuffle(x)

        return x


class Restormer(nn.Module):
    """Restormer model for image deblurring.
  
    Details regarding the model architecture can be found in the paper
    "Restormer: Efficient Transformer for High-Resolution Image Restoration".
    """

    def __init__(self, num_layers=4, num_transformer_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
                 channels=[48, 96, 192, 384], num_refinement_blocks=4, expansion_factor=2.66, bias=False):
        """Constructor of Restormer. Default values have set for each argument as per the paper.

        Args:
            num_layers: number of layers in Restormer.
            num_transformer_blocks: list containing number of transformer blocks
                in each layer for both the encoder and decoder .
            heads: list of number of attention heads in each layer of both the
                encoder and decoder.
            channels: list of number of input channels in input for each layer of
                both the encoder and decoder
            num_refinement_blocks: number of refinement transformer blocks used
            expansion_factor: expansion factor GDFN module of each Transformer
                block.
            bias: if True, each convolutional layer will have a bias in model.
        """

        super(Restormer, self).__init__()
        if num_layers < 2:
            raise Exception("Minimum number of layers is 2")
        if len(num_transformer_blocks) != num_layers:
            raise Exception("Length of list of num_transformer_block should be equal to num_layers which is: ",
                            num_layers)
        if len(heads) != num_layers:
            raise Exception("Length of list of heads should be equal to num_layers which is: ", num_layers)
        if len(channels) != num_layers:
            raise Exception("Length of list of channels should be equal to num_layers which is: ", num_layers)

        self.num_layers = num_layers
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=3, padding=1, bias=bias)
        self.output_conv = torch.nn.Conv2d(in_channels=channels[1], out_channels=3, kernel_size=3, padding=1, bias=bias)

        self.encoders = torch.nn.ModuleList()
        for i in range(num_layers):
            encoder_i = [
                TransformerBlock(channels=channels[i], num_heads=heads[i], expansion_factor=expansion_factor, bias=bias)
                for j in range(num_transformer_blocks[i])]
            self.encoders.append(torch.nn.Sequential(*encoder_i))

        self.decoders = torch.nn.ModuleList()
        for i in range(num_layers - 2, -1, -1):
            if i == 0:
                decoder_i = [
                    TransformerBlock(channels=channels[1], num_heads=heads[0], expansion_factor=expansion_factor,
                                     bias=bias) for j in range(num_transformer_blocks[i])]
            else:
                decoder_i = [
                    TransformerBlock(channels=channels[i], num_heads=heads[i], expansion_factor=expansion_factor,
                                     bias=bias) for j in range(num_transformer_blocks[i])]

            self.decoders.append(torch.nn.Sequential(*decoder_i))

        self.refinement_block = torch.nn.Sequential(
            *[TransformerBlock(channels=channels[1], num_heads=heads[0], expansion_factor=expansion_factor,
                               bias=bias) for i in range(num_refinement_blocks)])

        self.up_samples = torch.nn.ModuleList(UpSample(channels=i, bias=bias) for i in list(reversed(channels))[:-1])
        self.down_samples = torch.nn.ModuleList(DownSample(channels=i, bias=bias) for i in channels[:-1])

        channels_no_for_conv1 = list(reversed(channels))[:-1]
        self.conv1_reduces = torch.nn.ModuleList(
            torch.nn.Conv2d(in_channels=channels_no_for_conv1[i], out_channels=channels_no_for_conv1[i + 1],
                            kernel_size=1, bias=bias) for i in range(len(channels_no_for_conv1) - 1))

    def forward(self, x):
        embed_x = self.input_conv(x)
        encodings = []

        for i in range(self.num_layers):
            if i == 0:
                encodings.append(self.encoders[i](embed_x))
            else:
                down_sample_enc = self.down_samples[i - 1](encodings[i - 1])
                encodings.append(self.encoders[i](down_sample_enc))

        decoding = encodings[self.num_layers - 1]
        for i in range(self.num_layers - 1):
            if i == self.num_layers - 2:
                up_sample_dec = self.up_samples[i](decoding)
                dec_input = torch.cat([up_sample_dec, encodings[self.num_layers - i - 2]], dim=1)
                decoding = self.decoders[i](dec_input)
            else:
                up_sample_dec = self.up_samples[i](decoding)
                dec_input = self.conv1_reduces[i](torch.cat([up_sample_dec, encodings[self.num_layers - i - 2]], dim=1))
                decoding = self.decoders[i](dec_input)

        refined_dec = self.refinement_block(decoding)
        final_out = self.output_conv(refined_dec)

        return final_out + x
