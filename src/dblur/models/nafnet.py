import torch.nn as nn
import torch


class SimpleGate(nn.Module):
    def __init__(self, reduction_factor=2):
        super(SimpleGate, self).__init__()
        self.reduction_factor = reduction_factor

    def forward(self, x):
        x = x.chunk(self.reduction_factor, dim=1)
        out = x[0]
        for i in range(1, self.reduction_factor):
            out = out * x[i]

        return out


class SCA(nn.Module):
    def __init__(self, channels, bias=True):
        super(SCA, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
        )

    def forward(self, x):
        return self.sca(x)


class AttnBlock(nn.Module):
    def __init__(self, channels, attn_expansion_factor=2, gate_reduction_factor=2, dropout_rate=0, kernel_size=3,
                 bias=True):
        super(AttnBlock, self).__init__()
        attn_channels = channels * attn_expansion_factor
        self.project_in_conv = nn.Conv2d(in_channels=channels, out_channels=attn_channels, kernel_size=1, padding=0,
                                         stride=1, bias=bias)
        self.conv = nn.Conv2d(in_channels=attn_channels, out_channels=attn_channels, kernel_size=kernel_size,
                              padding="same", stride=1, groups=attn_channels, bias=bias)
        self.simple_gate = SimpleGate(reduction_factor=gate_reduction_factor)
        self.sca = SCA(attn_channels // gate_reduction_factor)
        self.project_out_conv = nn.Conv2d(in_channels=attn_channels // gate_reduction_factor, out_channels=channels,
                                          kernel_size=1, padding=0, stride=1, bias=bias)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()

    def forward(self, x):
        out = self.project_in_conv(x)
        out = self.conv(out)
        out = self.simple_gate(out)
        out = out * self.sca(out)
        out = self.project_out_conv(out)
        out = self.dropout(out)

        return out + x


class FFNBlock(nn.Module):
    def __init__(self, channels, ffn_expansion_factor=2, gate_reduction_factor=2, dropout_rate=0, bias=True):
        super(FFNBlock, self).__init__()
        ffn_channels = channels * ffn_expansion_factor
        self.project_in_conv = nn.Conv2d(in_channels=channels, out_channels=ffn_channels, kernel_size=1, padding=0,
                                         stride=1, bias=bias)
        self.simple_gate = SimpleGate(reduction_factor=gate_reduction_factor)
        self.project_out_conv = nn.Conv2d(in_channels=ffn_channels // gate_reduction_factor, out_channels=channels,
                                          kernel_size=1, padding=0, stride=1, bias=bias)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()

    def forward(self, x):
        out = self.project_in_conv(x)
        out = self.simple_gate(out)
        out = self.project_out_conv(out)
        out = self.dropout(out)

        return out + x


class NAFNetBlock(nn.Module):
    def __init__(self, channels, attn_expansion_factor=2, ffn_expansion_factor=2, gate_reduction_factor=2,
                 dropout_rate=0, bias=True):
        super(NAFNetBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(channels)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.attn_block = AttnBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                    gate_reduction_factor=gate_reduction_factor,
                                    dropout_rate=dropout_rate, bias=bias)
        self.ffn_block = FFNBlock(channels=channels, ffn_expansion_factor=ffn_expansion_factor,
                                  gate_reduction_factor=gate_reduction_factor,
                                  dropout_rate=dropout_rate, bias=bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = self.layer_norm1(x.reshape(batch_size, channels, height * width).transpose(1, 2))
        out = out.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = self.attn_block(out)

        out = self.layer_norm2(out.reshape(batch_size, channels, height * width).transpose(1, 2))
        out = out.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = self.ffn_block(out)

        return out


class DownSample(nn.Module):
    def __init__(self, channels, downscale_factor=2, kernel_size=1, bias=False):
        super(DownSample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels // downscale_factor,
                                    kernel_size=kernel_size, padding="same", bias=bias)
        self.px_unshuffle = torch.nn.PixelUnshuffle(downscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.px_unshuffle(x)

        return x


class UpSample(nn.Module):
    def __init__(self, channels, upscale_factor=2, kernel_size=1, bias=False):
        super(UpSample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels * upscale_factor,
                                    kernel_size=kernel_size, padding="same", bias=bias)
        self.px_shuffle = torch.nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.px_shuffle(x)

        return x


class NAFNet(nn.Module):
    """
    NAFNet model for image deblurring. 
    
    Details regarding the model architecture can be found in the paper "Simple 
    Baselines for Image Restoration".  
    """

    def __init__(self, projected_in_channels=32, enc_num=[1, 1, 1, 28], middle_num=1, dec_num=[1, 1, 1, 1],
                 attn_expansion_factor=2, ffn_expansion_factor=2, gate_reduction_factor=2, dropout_rate=0,
                 upscale_factor=2, attn_kernel_size=3, upscale_kernel_size=1, bias=True, upscale_bias=False):

        """Constructor for NAFNet. Default values have set for each argument as per the paper.
      
        Args:
            projected_in_channels: number of channels input is projected into by
                the first convolution layer.
            enc_num: list of number of NAFNet blocks in each encoder for a
                particular layer of the UNet.
            middle_num: number of NAFNet blocks in lowest layer of the UNet.
            dec_num: list of number of NAFNet blocks in each decoder for a
                particular layer of the UNet.
            attn_expansion_factor: expansion factor in the 1D Convolution in the
                attention block of each NAFNet Block.
            ffn_expansion_factor: expansion factor in the 1D Convolution in the
                Feed Forward Block of each NAFNet Block.
            gate_reduction_factor: reductor factor in the SimpleGate present in
                each Feed Forward Block of each NAFNet Block.
            dropout_rate: dropout rate in each NAFNet Block.
            upscale_factor: scaling factor across different layers in the UNet.
                E.g. With a scaling factor of 2, the image dimensions are halved
                as we go move downward across layers in the UNet.
            attn_kernel_size: kernel size of the convolutional layer in each
                attention block in each NAFNet Block.
            upscale_kernel_size: kernel size of the convolution layer in each
                UpSample and DownSample block.
            bias: if True, bias is added in every convolution layer in the model
                except for the convolutions present in the UpSample and DownSample
                blocks.
            upscale_bias: if True, bias is added in every convolution layer in the
                UpSample and Donwsample block.
        """

        super(NAFNet, self).__init__()
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=projected_in_channels, kernel_size=3, padding="same",
                                    bias=bias)
        self.output_conv = nn.Conv2d(in_channels=projected_in_channels, out_channels=3, kernel_size=3, padding="same",
                                     bias=bias)

        num_layers = len(enc_num)
        downscale_factor = upscale_factor
        downscale_kernel_size = upscale_kernel_size
        downscale_bias = upscale_bias
        self.encoders = torch.nn.ModuleList()
        self.down_samples = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.up_samples = torch.nn.ModuleList()

        channels = projected_in_channels
        for i in range(num_layers):
            encoder_i = [NAFNetBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate, bias=bias)
                         for j in range(enc_num[i])]
            self.encoders.append(torch.nn.Sequential(*encoder_i))
            self.down_samples.append(
                DownSample(channels=channels, downscale_factor=downscale_factor, kernel_size=downscale_kernel_size,
                           bias=downscale_bias))
            channels = channels * 2

        middle_block = [NAFNetBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate, bias=bias)
                        for j in range(middle_num)]
        self.middle_block = torch.nn.Sequential(*middle_block)

        for i in range(num_layers):
            self.up_samples.append(
                UpSample(channels=channels, upscale_factor=upscale_factor, kernel_size=upscale_kernel_size,
                         bias=upscale_bias))
            channels = channels // 2
            decoder_i = [NAFNetBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate, bias=bias)
                         for j in range(dec_num[i])]
            self.decoders.append(torch.nn.Sequential(*decoder_i))

    def forward(self, x):
        out = self.input_conv(x)
        encodings = []
        for i in range(len(self.encoders)):
            out = self.encoders[i](out)
            encodings.append(out)
            out = self.down_samples[i](out)

        out = self.middle_block(out)

        for i in range(len(self.decoders)):
            out = self.up_samples[i](out)
            out = out + encodings[len(encodings) - 1 - i]
            out = self.decoders[i](out)

        out = self.output_conv(out)

        return out + x
