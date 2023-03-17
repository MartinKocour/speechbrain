"""Library to support dual-path speech separation.

Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding

from speechbrain.lobes.models.dual_path import (
    select_norm,
    SBTransformerBlock,
    SBRNNBlock,
)

from copy import deepcopy

EPS = 1e-8

class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L] or [B, C, L],
            where C = number of audio channels
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, C, N, T_out].

        where B = Batchsize
              C = Number of channels
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if x.ndim == 2:
            B, _ = x.shape
            C = 1
            x = torch.unsqueeze(x, dim=1)
        else:
            B, C, _ = x.shape
            # [BC, 1, L]
            x = x.view(B*C, self.in_channels, -1)

        # BC x 1 x L -> BC x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        # [B, C, N, L]
        x = x.view(B, C, self.out_channels, -1)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L] or [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        assert x.dim() in [2, 3], "{} accept 2/3D tensor as input".format(self)
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Triple_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra models.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
        self,
        intra_channel_mdl,
        intra_chunk_mdl,
        inter_chunk_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super(Triple_Computation_Block, self).__init__()

        self.intra_channel_mdl = intra_channel_mdl
        self.intra_chunk_mdl = intra_chunk_mdl
        self.inter_chunk_mdl = inter_chunk_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_channel_norm = select_norm(norm, out_channels, 5)
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            if isinstance(intra_channel_mdl, SBRNNBlock):
                self.intra_channel_linear = Linear(
                    out_channels, input_size=2 * intra_channel_mdl.mdl.rnn.hidden_size
                )
            else:
                self.intra_channel_linear = Linear(
                    out_channels, input_size=out_channels
                )

            if isinstance(intra_chunk_mdl, SBRNNBlock):
                self.intra_linear = Linear(
                    out_channels, input_size=2 * intra_chunk_mdl.mdl.rnn.hidden_size
                )
            else:
                self.intra_linear = Linear(
                    out_channels, input_size=out_channels
                )

            if isinstance(inter_chunk_mdl, SBRNNBlock):
                self.inter_linear = Linear(
                    out_channels, input_size=2 * intra_chunk_mdl.mdl.rnn.hidden_size
                )
            else:
                self.inter_linear = Linear(
                    out_channels, input_size=out_channels
                )

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, C, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, C, N, K, S].
            where, B = Batchsize,
               C = number of audio channels
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, C, N, K, S = x.shape
        # intra channel
        # [BKS, C, N]
        intra_ch = x.permute(0, 3, 4, 1, 2).contiguous().view(B * K * S, C, N)

        # [BKS, C, H]
        intra_ch = self.intra_channel_mdl(intra_ch)

        # [BKS, C, N]
        if self.linear_layer_after_inter_intra:
            intra_ch = self.intra_channel_linear(intra_ch)

        # [B, K, S, C, N]
        intra_ch = intra_ch.view(B, K, S, C, N)

        # [B, N, K, S, C]
        intra_ch = intra_ch.permute(0, 4, 1, 2, 3).contiguous()
        if self.norm:
            intra_ch = self.intra_channel_norm(intra_ch)

        # [BCS, K, N]
        intra = intra_ch.permute(0, 4, 3, 2, 1).contiguous().view(B * C * S, K, N)

        # [BCS, K, H]
        intra = self.intra_chunk_mdl(intra)

        # [BCS, K, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        # [BC, S, K, N]
        intra = intra.view(B*C, S, K, N)
        # [BC, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, C, N, K, S]
        intra = intra.contiguous().view(B, C, N, K, S)
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BCK, S, N]
        inter = intra.permute(0, 1, 3, 4, 2).contiguous().view(B * C * K, S, N)
        # [BCK, S, H]
        inter = self.inter_chunk_mdl(inter)

        # [BCK, S, N]
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        # [BC, K, S, N]
        inter = inter.view(B*C, K, S, N)
        # [BC, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)

        # [B, C, N, K, S]
        inter = inter.contiguous().view(B, C, N, K, S)

        # [B, C, N, K, S]
        out = inter + intra

        return out


class Triple_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_channel_model,
        intra_chunk_model,
        inter_chunk_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Triple_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.tripple_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.tripple_mdl.append(
                copy.deepcopy(
                    Triple_Computation_Block(
                        intra_channel_model,
                        intra_chunk_model,
                        inter_chunk_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, C, N, L], 
            where C = number of audio channels (optional)

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        assert x.ndim == 4, "Expected 4D (batched) input to Triple_Path_Model, but got input of size: {}".format(x.shape)

        B, C, N, L = x.shape

        # [BC, N, L]
        x = x.contiguous().view(B*C, N, L)

        # [BC, N, L]
        x = self.norm(x)

        # [BC, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        # [BC, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        _, _, K, S = x.shape

        # [B, C, N, K, S]
        x = x.contiguous().view(B, C, N, K, S)

        # [B, C, N, K, S]
        for i in range(self.num_layers):
            x = self.tripple_mdl[i](x)
        x = self.prelu(x)

        # [B, N, K, S]
        # aggregate over audio channels
        # TODO: replace mean with something trainable, e.g. MHA
        x = x.mean(dim=1)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class MultiChannelSepformerWrapper(nn.Module):
    """The wrapper for the sepformer model which combines the Encoder, Masknet and the decoder
    https://arxiv.org/abs/2010.13154

    Arguments
    ---------

    encoder_kernel_size: int,
        The kernel size used in the encoder
    encoder_in_nchannels: int,
        The number of channels of the input audio
    encoder_out_nchannels: int,
        The number of filters used in the encoder.
        Also, number of channels that would be inputted to the intra and inter blocks.
    masknet_chunksize: int,
        The chunk length that is to be processed by the intra blocks
    masknet_numlayers: int,
        The number of layers of combination of inter and intra blocks
    masknet_norm: str,
        The normalization type to be used in the masknet
        Should be one of 'ln' -- layernorm, 'gln' -- globallayernorm
                         'cln' -- cumulative layernorm, 'bn' -- batchnorm
                         -- see the select_norm function above for more details
    masknet_useextralinearlayer: bool,
        Whether or not to use a linear layer at the output of intra and inter blocks
    masknet_extraskipconnection: bool,
        This introduces extra skip connections around the intra block
    masknet_numspks: int,
        This determines the number of speakers to estimate
    intra_numlayers: int,
        This determines the number of layers in the intra block
    inter_numlayers: int,
        This determines the number of layers in the inter block
    intra_nhead: int,
        This determines the number of parallel attention heads in the intra block
    inter_nhead: int,
        This determines the number of parallel attention heads in the inter block
    intra_dffn: int,
        The number of dimensions in the positional feedforward model in the inter block
    inter_dffn: int,
        The number of dimensions in the positional feedforward model in the intra block
    intra_use_positional: bool,
        Whether or not to use positional encodings in the intra block
    inter_use_positional: bool,
        Whether or not to use positional encodings in the inter block
    intra_norm_before: bool
        Whether or not we use normalization before the transformations in the intra block
    inter_norm_before: bool
        Whether or not we use normalization before the transformations in the inter block

    Example
    -----
    >>> model = SepformerWrapper()
    >>> inp = torch.rand(1, 160)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_norm="ln",
        masknet_useextralinearlayer=False,
        masknet_extraskipconnection=True,
        masknet_numspks=2,
        intra_channel_numlayers=8,
        intra_numlayers=8,
        inter_numlayers=8,
        intra_nhead=8,
        inter_nhead=8,
        intra_dffn=1024,
        inter_dffn=1024,
        intra_use_positional=True,
        inter_use_positional=True,
        intra_norm_before=True,
        inter_norm_before=True,
    ):

        super(FlexformerWrapper, self).__init__()
        self.encoder = Encoder(
            kernel_size=encoder_kernel_size,
            out_channels=encoder_out_nchannels,
            in_channels=encoder_in_nchannels,
        )
        intra_channel_mdl = SBTransformerBlock(
            num_layers=intra_channel_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before,
        )

        intra_chunk_mdl = SBTransformerBlock(
            num_layers=intra_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before,
        )

        inter_chunk_mdl = SBTransformerBlock(
            num_layers=inter_numlayers,
            d_model=encoder_out_nchannels,
            nhead=inter_nhead,
            d_ffn=inter_dffn,
            use_positional_encoding=inter_use_positional,
            norm_before=inter_norm_before,
        )

        self.masknet = Triple_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_channel_model=intra_channel_mdl,
            intra_chunk_model=intra_chunk_mdl,
            inter_chunk_model=inter_chunk_mdl,
            num_layers=masknet_numlayers,
            norm=masknet_norm,
            K=masknet_chunksize,
            num_spks=masknet_numspks,
            skip_around_intra=masknet_extraskipconnection,
            linear_layer_after_inter_intra=masknet_useextralinearlayer,
        )
        self.decoder = Decoder(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_in_nchannels,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            bias=False,
        )
        self.num_spks = masknet_numspks

        # reinitialize the parameters
        for module in [self.encoder, self.masknet, self.decoder]:
            self.reset_layer_recursively(module)

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the network"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def forward(self, mix):
        """ Processes the input tensor x and returns an output tensor."""
        # [B, C, N, L]
        mix_w = self.encoder(mix)
        # [spks, B, N, L]
        est_mask = self.masknet(mix_w)
        # [spks, B, N, L]
        mix_w = torch.stack([mix_w] * self.num_spks)
        # [spks, B, N, L]
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source
