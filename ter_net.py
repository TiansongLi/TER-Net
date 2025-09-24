import torch.nn.init as init
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from thop import profile, clever_format
from networks.utils.new_graph_module import GCN
from networks.utils.wave import EnhancedHighFreqBlock
from networks.utils.DSConv import DSConv, SE_Block
from networks.utils.cbam import ChannelAttention2D, SpatialAttention2D
from networks.utils.multi import DirectionalConvModule


class MemoryEfficientMish(nn.Module):
    # Mish activation memory-efficient
    class F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class down_pooling(nn.Module):
    def __init__(self, ch):
        super(down_pooling, self).__init__()
        self.down = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.down(x)
        return x


class conv_block0(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block0, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        return x3


class conv_block1(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block1, self).__init__()
        self.conv1 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2


class conv_block2(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block2, self).__init__()
        self.conv1 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv4 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        x4 = self.conv4(x3 + x2 + x1)
        return x4


class conv_standard(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super(conv_standard, self).__init__()
        self.doudouble_conv = nn.Sequential(
            conv_block0(in_ch, middle_ch),
            conv_block2(middle_ch, middle_ch),
            conv_block2(middle_ch, out_ch),
        )

    def forward(self, x):
        x = self.doudouble_conv(x)

        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class fuconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fuconv, self).__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch // 4, 3, 2, 1)
        self.down = nn.MaxPool2d(4)
        self.conv_2 = nn.Conv2d(out_ch // 4, out_ch, 3, 2, 1)
        self.conv_3 = nn.Conv2d(out_ch, out_ch // 4, 1)

        self.conv = nn.Conv2d(out_ch // 2, out_ch, 1)
        self.dsconv = DSConv(out_ch, out_ch, 1, 1, 0, True, 'cuda:1')
        self.se = SE_Block(out_ch)

    def forward(self, x_ske, x_enc):
        res = x_enc
        x_ske = self.conv_1(x_ske)
        x_ske = self.down(x_ske)
        x_ske = self.conv_2(x_ske)
        # x_enc = self.conv_3(x_enc)

        x_fu = x_ske + x_enc
        # x = self.conv(x_fu)
        # x = self.se(x)
        # x = x + res
        # x_out = self.dsconv(x)
        return x_fu


class extra_fea(nn.Module):
    def __init__(self, in_channels):
        super(extra_fea, self).__init__()

        self.ca = ChannelAttention2D(in_channels)
        self.sa = SpatialAttention2D()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.dsc1 = DSConv(in_channels, out_ch=in_channels, kernel_size=1, extend_scope=1, morph=0, if_offset=True, device='cuda:1')
        # self.dsc = DSConv(in_channels, out_ch=in_channels // 2, kernel_size=1, extend_scope=1, morph=0, if_offset=True,
        #                   device='cuda')

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.PReLU()
        self.out = outconv(in_channels, 1)

    def forward(self, x):
        res = x

        x_ca = self.ca(x)
        x_sa = self.sa(x)

        # fuse = torch.cat([x_ca, x_sa], dim=1)
        fuse = x_sa + x_ca
        guide = self.gate_conv(fuse)  # [B, C, H, W]

        x = x * guide
        x = self.dsc1(x)

        x = x + res
        # x = self.dsc(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.out(x)

        return x


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # self.conv1 = conv_block0(1, 32)
        # self.conv2 = DirectionalConvModule(32, 64)
        # self.conv3 = DirectionalConvModule(64, 128)
        # self.conv4 = DirectionalConvModule(128, 128 * 2)
        # self.conv5 = DirectionalConvModule(128 * 2, 128 * 4)
        self.conv1 = conv_block1(1, 32)
        self.conv2 = conv_block1(32, 64)
        self.conv3 = conv_block1(64, 128)
        self.conv4 = conv_block1(128, 128 * 2)
        self.conv5 = conv_block1(128 * 2, 128 * 4)

        self.pool32 = down_pooling(32)
        self.pool64 = down_pooling(64)
        self.pool128 = down_pooling(128)
        self.pool256 = down_pooling(256)

    def forward(self, x):
        # share encoder
        x1 = self.conv1(x)
        x2 = self.pool32(x1)
        x2 = self.conv2(x2)
        x3 = self.pool64(x2)
        x3 = self.conv3(x3)
        x4 = self.pool128(x3)
        x4 = self.conv4(x4)
        x5 = self.pool256(x4)
        x5 = self.conv5(x5)  # 1,512,32,32

        return x1, x2, x3, x4, x5


class seg_deconder(nn.Module):
    def __init__(self):
        super(seg_deconder, self).__init__()
        self.seg_upconv1 = self.upconv(64, 32)
        self.seg_upconv2 = self.upconv(128, 64)
        self.seg_upconv3 = self.upconv(256, 128)
        self.seg_upconv4 = self.upconv(128 * 4, 128 * 2)

        self.fuconv = fuconv(1, 512)

        self.seg_conv4 = DirectionalConvModule(128 * 4, 128 * 2)
        self.seg_conv3 = DirectionalConvModule(256, 128)
        self.seg_conv2 = DirectionalConvModule(128, 64)
        self.seg_conv1 = DirectionalConvModule(64, 32)

        # self.seg_conv4 = conv_block1(128 * 4, 128 * 2)
        # self.seg_conv3 = conv_block1(256, 128)
        # self.seg_conv2 = conv_block1(128, 64)
        # self.seg_conv1 = conv_block1(64, 32)

        self.outc2 = outconv(32, 1)

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)

    def forward(self, pre_ske, x1, x2, x3, x4, x5, ske4, ske3, ske2, ske1):
        # seg branch
        #  s4 = self.upconv()
        x_seg = self.fuconv(pre_ske, x5)  # 1, 512, 32, 32
        seg_4 = self.seg_upconv4(x_seg * ske4)
        seg_4 = self.seg_conv4(torch.cat([seg_4, x4], dim=1))

        seg_3 = self.seg_upconv3(seg_4 * ske3)
        seg_3 = self.seg_conv3(torch.cat([seg_3, x3], dim=1))

        seg_2 = self.seg_upconv2(seg_3 * ske2)
        seg_2 = self.seg_conv2(torch.cat([seg_2, x2], dim=1))

        seg_1 = self.seg_upconv1(seg_2 * ske1)
        seg_1 = self.seg_conv1(torch.cat([seg_1, x1], dim=1))
        pre_seg = self.outc2(seg_1)

        return pre_seg, seg_1


class ske_decoder(nn.Module):
    def __init__(self):
        super(ske_decoder, self).__init__()
        self.ske_upconv1 = self.upconv(64, 32)
        self.ske_upconv2 = self.upconv(128, 64)
        self.ske_upconv3 = self.upconv(256, 128)
        self.ske_upconv4 = self.upconv(128 * 4, 128 * 2)

        self.GCN = GCN(512)

        self.fre4 = EnhancedHighFreqBlock(512)
        self.fre3 = EnhancedHighFreqBlock(256)
        self.fre2 = EnhancedHighFreqBlock(128)
        self.fre1 = EnhancedHighFreqBlock(64)

        self.ske_conv4 = conv_block1(128 * 4, 128 * 2)
        self.ske_conv3 = conv_block1(256, 128)
        self.ske_conv2 = conv_block1(128, 64)
        self.ske_conv1 = conv_block1(64, 32)

        self.outc1 = outconv(32, 1)

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)

    def forward(self, x1, x2, x3, x4, x5):
        # ske branch
        x_ske = self.GCN(x5)
        ske4 = self.fre4(x_ske)
        ske_4 = self.ske_upconv4(ske4)
        ske_4 = self.ske_conv4(torch.cat([ske_4, x4], dim=1))  # 1, 256, 64, 64

        ske3 = self.fre3(ske_4)
        ske_3 = self.ske_upconv3(ske3)
        ske_3 = self.ske_conv3(torch.cat([ske_3, x3], dim=1))  # 1, 128, 128, 128

        ske2 = self.fre2(ske_3)
        ske_2 = self.ske_upconv2(ske2)
        ske_2 = self.ske_conv2(torch.cat([ske_2, x2], dim=1))  # 1, 64, 256, 256

        ske1 = self.fre1(ske_2)
        ske_1 = self.ske_upconv1(ske1)
        ske_1 = self.ske_conv1(torch.cat([ske_1, x1], dim=1))  # 1, 32, 512, 512
        ske = self.outc1(ske_1)
        pre_ske = F.sigmoid(ske)

        return ske, pre_ske, ske4, ske3, ske2, ske1, ske_1


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc = encoder()
        self.ske_dec = ske_decoder()
        self.seg_dec = seg_deconder()
        self.final = extra_fea(32)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.enc(x)  # 1,512,32,32

        # ske branch
        ske, pre_ske, ske4, ske3, ske2, ske1, ske_1 = self.ske_dec(x1, x2, x3, x4, x5)

        # seg branch
        pre_seg1, seg_1 = self.seg_dec(pre_ske, x1, x2, x3, x4, x5, ske4, ske3, ske2, ske1)

        pre_seg = self.final(seg_1+ske_1)

        return ske, pre_seg, pre_seg1


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 512, 512).to(device)
    model = UNet().to(device)
    print('ske:', model(x)[0].shape)
    print('pre_seg:', model(x)[1].shape)
    flops, params = profile(model, inputs=(x,), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    print(f"运算量:{flops}, 参数量:{params}")




