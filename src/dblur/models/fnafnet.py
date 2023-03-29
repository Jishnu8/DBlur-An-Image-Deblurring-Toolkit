import torch.nn as nn
import torch
from dblur.models.nafnet import AttnBlock, FFNBlock, UpSample, DownSample


class ResFFTBlock(nn.Module):
    def __init__(self, channels, expansion_factor, fft_norm="backward", activation_func=nn.ReLU(), bias=False, a=4):
        super(ResFFTBlock, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.activation = activation_func
        self.bias = bias
        self.fft_norm = fft_norm

        hidden_channels = channels * expansion_factor
        self.complex_weight1 = torch.complex(nn.Parameter(torch.Tensor(channels, hidden_channels)).to(device),
                                             nn.Parameter(torch.Tensor(channels, hidden_channels)).to(device))
        self.complex_weight2 = torch.complex(nn.Parameter(torch.Tensor(hidden_channels, channels)).to(device),
                                             nn.Parameter(torch.Tensor(hidden_channels, channels)).to(device))
        if bias:
            self.complex_bias1 = torch.complex(nn.Parameter(torch.zeros((1, 1, 1, hidden_channels)).to(device)),
                                               nn.Parameter(torch.zeros((1, 1, 1, hidden_channels))).to(device))
            self.complex_bias2 = torch.complex(nn.Parameter(torch.zeros((1, 1, 1, channels)).to(device)),
                                               nn.Parameter(torch.zeros((1, 1, 1, channels))).to(device))

        self.init_weights(a=a)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        z = torch.fft.rfft2(x, norm=self.fft_norm)
        batch_size_z, channels_z, height_z, width_z = z.shape
        z = z.reshape(batch_size_z, channels_z, height_z * width_z).transpose(1, 2).reshape(batch_size_z, height_z,
                                                                                            width_z, channels_z)
        z = z @ self.complex_weight1
        if self.bias:
            z = z + self.complex_bias1

        z.real = self.activation(z.real)
        z.imag = self.activation(z.imag)
        z = z @ self.complex_weight2
        if self.bias:
            z = z + self.complex_bias2

        z = z.reshape(batch_size_z, height_z * width_z, channels_z).transpose(1, 2).reshape(batch_size_z, channels_z,
                                                                                            height_z, width_z)
        z = torch.fft.irfft2(z, s=(height, width), norm=self.fft_norm)

        return z

    def init_weights(self, a=4):
        nn.init.kaiming_uniform_(self.complex_weight1.real, a=a)
        nn.init.kaiming_uniform_(self.complex_weight1.imag, a=a)
        nn.init.kaiming_uniform_(self.complex_weight2.real, a=a)
        nn.init.kaiming_uniform_(self.complex_weight2.imag, a=a)


class FNAFNetBlock(nn.Module):
    def __init__(self, channels, attn_expansion_factor=2, ffn_expansion_factor=2, gate_reduction_factor=2,
                 dropout_rate=0, bias=True,
                 fft_block_expansion_factor=2, fft_block_norm="backward", fft_block_activation=nn.ReLU(),
                 fft_block_bias=False, fft_block_a=4):
        super(FNAFNetBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(channels)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.attn_block = AttnBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                    gate_reduction_factor=gate_reduction_factor,
                                    dropout_rate=dropout_rate, bias=bias)
        self.ffn_block = FFNBlock(channels=channels, ffn_expansion_factor=ffn_expansion_factor,
                                  gate_reduction_factor=gate_reduction_factor,
                                  dropout_rate=dropout_rate, bias=bias)

        self.res_fft_block = ResFFTBlock(channels=channels, expansion_factor=fft_block_expansion_factor,
                                         fft_norm=fft_block_norm,
                                         activation_func=fft_block_activation, bias=fft_block_bias, a=fft_block_a)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = self.layer_norm1(x.reshape(batch_size, channels, height * width).transpose(1, 2))
        out = out.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = self.attn_block(out)

        out = self.layer_norm2(out.reshape(batch_size, channels, height * width).transpose(1, 2))
        out = out.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = self.ffn_block(out)

        res_fft_out = self.res_fft_block(x)

        return out + res_fft_out


class FNAFNet(nn.Module):
    """
    FNAFNet model for image deblurring. 
    
    Details regarding the model architecture can be found in the paper 
    "Intriguing Findings of Frequency Selection for Image Deblurring".
    """

    def __init__(self, projected_in_channels=32, enc_num=[1, 1, 1, 28], middle_num=1, dec_num=[1, 1, 1, 1],
                 attn_expansion_factor=2, ffn_expansion_factor=2, gate_reduction_factor=2, dropout_rate=0,
                 upscale_factor=2, attn_kernel_size=3, upscale_kernel_size=1, bias=True, upscale_bias=False,
                 fft_block_expansion_factor=2, fft_block_norm="backward", fft_block_activation=nn.ReLU(),
                 fft_block_bias=False, fft_block_a=4):

        """Constructor for FNAFNet.

        Default values have set for each argument as per the paper.
      
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
            fft_block_expansion_factor: expansion factor of the 1D convolution
                present in each ResFFTBlock.
            fft_block_norm: norm of the inverse 2D Fourier Transform present in
                each ResFFTBlock.
            fft_block_activation: activation function present in each ResFFTBlock.
            fft_block_bias: if True, bias is added in the 1D convolution present
                in each ResFFTBlock.
            fft_block_a: the negative slope of the rectifier used after this layer
                for the initialization of weights using
                torch.nn.init.kaiming_uniform.
        """

        super(FNAFNet, self).__init__()
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
            encoder_i = [FNAFNetBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate, bias=bias,
                                      fft_block_expansion_factor=fft_block_expansion_factor,
                                      fft_block_norm=fft_block_norm,
                                      fft_block_activation=fft_block_activation, fft_block_bias=fft_block_bias,
                                      fft_block_a=fft_block_a) for j in range(enc_num[i])]
            self.encoders.append(torch.nn.Sequential(*encoder_i))
            self.down_samples.append(
                DownSample(channels=channels, downscale_factor=downscale_factor, kernel_size=downscale_kernel_size,
                           bias=downscale_bias))
            channels = channels * 2

        middle_block = [FNAFNetBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate, bias=bias,
                                     fft_block_expansion_factor=fft_block_expansion_factor,
                                     fft_block_norm=fft_block_norm,
                                     fft_block_activation=fft_block_activation, fft_block_bias=fft_block_bias,
                                     fft_block_a=fft_block_a) for j in range(middle_num)]
        self.middle_block = torch.nn.Sequential(*middle_block)

        for i in range(num_layers):
            self.up_samples.append(
                UpSample(channels=channels, upscale_factor=upscale_factor, kernel_size=upscale_kernel_size,
                         bias=upscale_bias))
            channels = channels // 2
            decoder_i = [FNAFNetBlock(channels=channels, attn_expansion_factor=attn_expansion_factor,
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      gate_reduction_factor=gate_reduction_factor, dropout_rate=dropout_rate, bias=bias,
                                      fft_block_expansion_factor=fft_block_expansion_factor,
                                      fft_block_norm=fft_block_norm,
                                      fft_block_activation=fft_block_activation, fft_block_bias=fft_block_bias,
                                      fft_block_a=fft_block_a) for j in range(dec_num[i])]
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
