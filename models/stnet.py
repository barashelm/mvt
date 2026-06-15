from collections import OrderedDict
import torch
import torch.nn as nn
from models.quantization import hquan, muquan, pquan, tquan
from utils.parser import args

__all__ = ["stnet"]


img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
img_size = 32
num_heads = 4 # for multi-head attention
depth = 1  # No of STB
qkv_bias = True
window = 8  # window size for LSA

class GroupAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super(GroupAttention, self).__init__()

        self.num_heads = num_heads
        head_dim = img_size // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(img_size, img_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(img_size, img_size)
        self.ws = window

    def forward(self, x):
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, C, h_group, self.ws, W)
        qkv = self.qkv(x).reshape(B, C, total_groups, -1, 3, self.num_heads, self.ws // self.num_heads).permute(4, 0, 1, 2, 5, 3, 6)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        return x

class GlobalAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super().__init__()

        self.dim = img_size
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim//window, self.dim//window * 2, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        self.sr = nn.Conv2d(2, 2, kernel_size=window, stride=window)
        self.norm = nn.LayerNorm(self.dim//window)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1, self.dim//window, self.dim//window).permute(0,1,3,2,4)
        x_ = self.sr(x).reshape(B, C, -1, self.dim//window, self.dim//window)
        x_ = self.norm(x_)
        kv = self.kv(x_).reshape(B, C, -1, 2, self.dim//window, self.dim//window).permute(3,0,1,4,2,5)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)

        return x

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.cc1 = nn.Linear(img_size, img_size)
        self.cc2 = nn.Linear(img_size, img_size)
        self.act = nn.GELU()

    def forward(self, x):

        x = self.cc1(x)
        x = self.act(x)
        x = self.cc2(x)

        return x


class WTL(nn.Module):
    def __init__(self, num_heads, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(img_size, eps=1e-6)
        self.attn1 = GroupAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.attn2 = GlobalAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm4 = nn.LayerNorm(img_size, eps=1e-6)
        self.mlp1 = MLP()
        self.mlp2 = MLP()

    def forward(self, x):

        x = x + self.attn1(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.mlp2(self.norm4(x))

        return x

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class hsigmoid(nn.Module):
    def forward(self, x):
        out = nn.functional.relu6(x + 3, inplace=True) / 6
        return out

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            img_size=img_size,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            reduction=4
    ):
        super().__init__()


        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.conv1 = nn.Conv2d(2,16, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(16,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(2*img_size*img_size, 2048 // reduction)

    def forward(self, x):

        n_samples = x.shape[0]
        x = self.conv1(x)
        x = self.conv5(x)
        X = x 

        for block in self.blocks:
            x = block(x)
        x = self.norm3(x)
        x = self.convT(x) 
        x = X + self.conv4(x)
        x = self.norm2(x)
        x = x.reshape(n_samples,2*img_size*img_size)
        x = self.fc(x)
        return x


class Decoder(nn.Module):   
    def __init__(self, reduction=4):
        super(Decoder, self).__init__()

        self.act = nn.Sigmoid()
        self.conv5 = nn.Conv2d(2,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)

        self.dense_layers = nn.Sequential(
            nn.Linear(2048 // reduction, img_total)
        )

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)

    def forward(self, x):
        img = self.dense_layers(x)
        img = img.view(-1, img_channels, img_height, img_width)

        out = self.decoder_feature(img)
        x = self.conv5(img)

        for block in self.blocks:
            x = block((x+out))

        x = self.norm2(x)
        x = self.convT(x)
        x = self.conv4(x) 

        for block in self.blocks:
            x = block((x+out))

        x = self.norm3(x)

        x = self.act(x) 

        return x


class STNet(nn.Module):
    def __init__(self, reduction=4):
        super(STNet, self).__init__()
        self.encoder = Encoder(depth=depth, num_heads=num_heads, qkv_bias=qkv_bias, reduction=reduction)
        self.decoder = Decoder(reduction=reduction)

    def forward(self, x):
        x = self.encoder(x)
        
        x = self.decoder(x)
        return x


def stnet(reduction=4):
    model = STNet(reduction=reduction)
    return model