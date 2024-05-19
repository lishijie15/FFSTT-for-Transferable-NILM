import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import loralib as lora
from timm.models.layers import DropPath, trunc_normal_
import math

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class SVT_channel_token_mixing(nn.Module):
    def __init__(self, dim, d_modelup):
        super().__init__()

        self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True)

        self.hidden_size = dim // 2
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.token_blocks = 4  # 原始是28
        self.class_blocks = d_modelup // 16  #16=2*8
        self.con_blocks = d_modelup // 8

        self.complex_weight_ll = nn.Parameter(torch.randn(dim // 2, 8, 8, dtype=torch.float32) * 0.02)
        self.complex_weight_ll_de = nn.Parameter(torch.randn(dim // 2, self.con_blocks, 8, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        self.complex_weight_lh_1_t = nn.Parameter(torch.randn(2, self.class_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_2_t = nn.Parameter(torch.randn(2, self.class_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b1_t = nn.Parameter(torch.randn(2, self.class_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b2_t = nn.Parameter(torch.randn(2, self.class_blocks, self.token_blocks, dtype=torch.float32) * 0.02)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink = 0.0  # args.fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.permute(x, (0, 3, 1, 2)).contiguous()  # (B, H, W, C) -> (B, C, H, W)
        B, C, H, W = x.shape  # this shape is required for dwt

        x1, x2 = torch.chunk(x, 2, dim=1)
        # x1 = self.conv(x1)

        x2 = x2.to(torch.float32)
        B, C1, a, b = x2.shape

        xl, xh = self.xfm(x2)
        if H == 8:
            xl = xl * self.complex_weight_ll
        else:
            xl = xl * self.complex_weight_ll_de

        xh[0] = torch.permute(xh[0], (5, 0, 2, 3, 4, 1)).contiguous()  #(B,C,k,H,W,2) 2是实部和虚部
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4],
                              self.num_blocks, self.block_size)  #block是为了进行通道混合，所以对通道数32进行了拆分，变成了4和8

        ###########################################################################################
        # This is for Channel mixing:
        x_real = xh[0][0]
        x_imag = xh[0][1]

        # print(x_real.shape, x_imag.shape, self.complex_weight_lh_1[0].shape, self.complex_weight_lh_1[1].shape, self.complex_weight_lh_2[0].shape, self.complex_weight_lh_2[1].shape)

        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])

        ###########################################################################################
        # This is for Token mixing:
        xh[0] = torch.permute(xh[0], (5, 0, 4, 1, 2, 3)).contiguous()  # 2, B, 64, 6,28,28
        x_real_t = xh[0][0]
        x_imag_t = xh[0][1]

        x_real_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[0]) - self.multiply(x_imag_t, self.complex_weight_lh_1_t[1]) + self.complex_weight_lh_b1_t[0])
        x_imag_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[1]) + self.multiply(x_imag_t, self.complex_weight_lh_1_t[0]) + self.complex_weight_lh_b1_t[1])
        x_real_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[0]) - self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[1]) + self.complex_weight_lh_b2_t[0]
        x_imag_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[1]) + self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[0]) + self.complex_weight_lh_b2_t[1]

        # print(x_real_t.shape, x_imag_t.shape, self.complex_weight_lh_1_t[0].shape, self.complex_weight_lh_1_t[1].shape, self.complex_weight_lh_2_t[0].shape, self.complex_weight_lh_2_t[1].shape)

        xh[0] = torch.stack([x_real_2_t, x_imag_2_t], dim=-1).float()  # B, 64, 6,24,28, 2

        x2 = self.ifm((xl, xh))

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C1, a, b)
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = x.reshape(B, N, C)  # permute is not same as reshape or view
        return x

# class SVT_channel_token_mixing_de(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#
#         self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True)
#
#         self.hidden_size = dim // 2
#         self.num_blocks = 4
#         self.block_size = self.hidden_size // self.num_blocks
#         assert self.hidden_size % self.num_blocks == 0
#         self.token_blocks = 4  # 原始是28
#
#         self.complex_weight_ll_de = nn.Parameter(torch.randn(dim // 2, 16, 8, dtype=torch.float32) * 0.02)
#         self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
#         self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
#         self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
#         self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
#
#         self.complex_weight_lh_1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
#         self.complex_weight_lh_2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
#         self.complex_weight_lh_b1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
#         self.complex_weight_lh_b2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
#
#         self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
#         self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
#         self.softshrink = 0.0  # args.fno_softshrink
#
#     def multiply(self, input, weights):
#         return torch.einsum('...bd,bdk->...bk', input, weights)
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.view(B, H, W, C)
#         x = torch.permute(x, (0, 3, 1, 2)).contiguous()  # (B, H, W, C) -> (B, C, H, W)
#         B, C, H, W = x.shape  # this shape is required for dwt
#
#         x1, x2 = torch.chunk(x, 2, dim=1)
#         # x1 = self.conv(x1)
#
#         x2 = x2.to(torch.float32)
#         B, C1, a, b = x2.shape
#
#         xl, xh = self.xfm(x2)
#         xl = xl * self.complex_weight_ll_de
#
#         xh[0] = torch.permute(xh[0], (5, 0, 2, 3, 4, 1)).contiguous()  #(B,C,k,H,W,2)
#         xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4],
#                               self.num_blocks, self.block_size)  #block是为了进行通道混合，所以对通道数32进行了拆分，变成了4和8
#
#         ###########################################################################################
#         # This is for Channel mixing:
#         x_real = xh[0][0]
#         x_imag = xh[0][1]
#
#         # print(x_real.shape, x_imag.shape, self.complex_weight_lh_1[0].shape, self.complex_weight_lh_1[1].shape, self.complex_weight_lh_2[0].shape, self.complex_weight_lh_2[1].shape)
#
#         x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
#         x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
#         x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
#         x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]
#
#         xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
#         xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
#         xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
#
#         ###########################################################################################
#         # This is for Token mixing:
#         xh[0] = torch.permute(xh[0], (5, 0, 4, 1, 2, 3)).contiguous()  # 2, B, 64, 6,28,28
#         x_real_t = xh[0][0]
#         x_imag_t = xh[0][1]
#
#         x_real_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[0]) - self.multiply(x_imag_t, self.complex_weight_lh_1_t[1]) + self.complex_weight_lh_b1_t[0])
#         x_imag_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[1]) + self.multiply(x_imag_t, self.complex_weight_lh_1_t[0]) + self.complex_weight_lh_b1_t[1])
#         x_real_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[0]) - self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[1]) + self.complex_weight_lh_b2_t[0]
#         x_imag_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[1]) + self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[0]) + self.complex_weight_lh_b2_t[1]
#
#         # print(x_real_t.shape, x_imag_t.shape, self.complex_weight_lh_1_t[0].shape, self.complex_weight_lh_1_t[1].shape, self.complex_weight_lh_2_t[0].shape, self.complex_weight_lh_2_t[1].shape)
#
#         xh[0] = torch.stack([x_real_2_t, x_imag_2_t], dim=-1).float()  # B, 64, 6,24,28, 2
#
#         x2 = self.ifm((xl, xh))
#
#         x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C1, a, b)
#         x = torch.permute(x, (0, 2, 3, 1)).contiguous()  # (N, C, H, W) -> (N, H, W, C)
#         x = x.reshape(B, N, C)  # permute is not same as reshape or view
#         return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self,
        dim,
        d_modelup,
        mlp_ratio,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = SVT_channel_token_mixing (dim, d_modelup)
        # self.attn_de = SVT_channel_token_mixing_de (dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if x.shape[1] == 64:
            W = H = 8
            x = x + self.drop_path(self.attn(self.norm1(x.transpose(-1, 1)), H, W))
        elif x.shape[1] == 128:
            W = 8
            H = 16
            x = x + self.drop_path(self.attn(self.norm1(x.transpose(-1, 1)), H, W))
        else:
            W = 8
            H = x.shape[1]//8
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

def make_dmodel(seq_len):
    remainder = seq_len // 8
    if remainder % 2 != 0:  # 如果余数不是偶数
        remainder = remainder + 1
        seq_len = remainder * 8
    elif remainder * 8 < seq_len:
        remainder = remainder + 2
        seq_len = remainder * 8
    return seq_len

class class_Encoder(nn.Module):
    def __init__(self, attention, d_model, seq_len, d_ff=None, dropout=0.1, activation="relu"):
        super(class_Encoder, self).__init__()
        d_ff = d_ff or 4 * d_model
        d_modelup = make_dmodel(seq_len)
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1 = lora.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, r=4)
        self.conv2 = lora.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, r=4)
        self.cell = Block(dim=d_model, d_modelup=d_modelup, mlp_ratio=0.4)
        # self.proj = nn.Linear(seq_len, d_modelup)
        # self.proj_b = nn.Linear(d_modelup, seq_len)
        self.proj = lora.Linear(seq_len, d_modelup, r=16)  #405, 416
        self.proj_b = lora.Linear(d_modelup, seq_len, r=16)  #416, 405
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.cell(self.proj(y.transpose(-1, 1)).transpose(-1, 1))
        y = self.proj_b(y.transpose(-1, 1))

        return self.norm2(x + y.transpose(-1, 1)), attn


class SVT_Encoder(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(SVT_Encoder, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.cell = Block(dim = d_model, mlp_ratio = 0.4)
        self.proj = nn.Linear(3, d_model)
        self.proj_b = nn.Linear(d_model, 3)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.cell(self.proj(y.transpose(-1, 1)))
        y = self.proj_b(y.transpose(-1, 1))

        return self.norm2(x + y.transpose(-1, 1)), attn

class SVT_Decoder(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(SVT_Decoder, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.cell = Block(dim = d_model, mlp_ratio = 0.4)
        self.proj = nn.Linear(11, d_model)
        self.proj_b = nn.Linear(d_model, 11)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.cell(self.proj(y.transpose(-1, 1)))
        y = self.proj_b(y.transpose(-1, 1))

        return self.norm2(x + y.transpose(-1, 1)), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i==0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

