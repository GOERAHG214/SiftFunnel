import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import dct, idct
from sympy.solvers.diophantine.diophantine import reconstruct

# Translated comment GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfAttention(nn.Module):
    def __init__(self, channels, size, num_head=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, num_head, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).contiguous().swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).contiguous().view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     nn.Conv2d(in_channels, out_channels // 4, 3, 1, 1),
        #     nn.SiLU(True),
        #     nn.Conv2d(out_channels // 4, out_channels // 4, 3, 1, 1),
        #     nn.Conv2d(out_channels // 4, out_channels, 3, 1, 1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(True),
        # )

    def forward(self, x):
        x = self.conv(x)
        return x


def apply_dct(x):
    x = x.detach().cpu().numpy()
    x_dct = dct(dct(x, axis=-1, norm='ortho'), axis=-2, norm='ortho')
    return torch.tensor(x_dct, device=device, dtype=torch.float32)


def apply_idct(x):
    x = x.detach().cpu().numpy()  # Translated commentCPU
    x_idct = idct(idct(x, axis=-1, norm='ortho'), axis=-2, norm='ortho')
    return torch.tensor(x_idct, device=device, dtype=torch.float32)


def low_freq_diagonal_mask(H, W):
    mask = np.zeros((H, W))
    threshold = (H + W) * 0.8
    for i in range(H):
        for j in range(W):
            if i + j < (H + W) / 2:
                mask[i, j] = 1
    return torch.tensor(mask, device=device, dtype=torch.float32)



def apply_low_freq_filter(dct_data):
    _, _, H, W = dct_data.shape
    mask = low_freq_diagonal_mask(H, W)
    filtered_data = dct_data * mask
    return filtered_data


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class pv4_classifier_4(nn.Module):
    def __init__(self, nc, ndf, nz, img_size, block_idx=0):
        super(pv4_classifier_4, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.img_size = img_size

        # Define blocks and layers (same as before)
        self.block1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )

        # Add SE module after block2_mix
        self.se_block1 = SEModule(ndf, reduction=32)

        self.triplet = TripletAttention()

        # self.block2_mix = nn.Sequential(
        #     nn.Conv2d(ndf, 12, 3, 1, 1),
        #     nn.BatchNorm2d(12),
        #     nn.MaxPool2d(2, 2, 0),
        #     nn.ReLU(True),
        # )
        self.block2_mix = nn.Sequential(
            nn.Conv2d(ndf, ndf // 4, 1, 1, 0),
            nn.BatchNorm2d(ndf // 4),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(ndf // 4, 12, 1, 1, 0),
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.SiLU(True),
            nn.Conv2d(12,12,1,1,0),
            nn.Conv2d(12,12,3,1,1),
            nn.Conv2d(12, 12, 5, 1, 2),
            nn.ReLU(True),
        )

        # Add SE module after block2_mix
        self.se_block2 = SEModule(12, reduction=4)

        self.cbam = CBAM(12, ratio=6)
        # self.sa2 = SelfAttention(12, 16, num_head=2)

        # self.triplet_edge = TripletAttention()

        self.up = Up(12, ndf * 2)
        # self.sa = SelfAttention(ndf * 2, 16)
        self.triplet_cloud = TripletAttention()

        self.block3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * int((self.img_size / 16)) * int((self.img_size / 16)), nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )


    def freeze_until_up(self):
        """Freezes all parameters in the network until the `up` layer."""
        for name, param in self.named_parameters():
            if name.startswith("up"):
                break
            param.requires_grad = False


    def forward(self, x, feat_out=None, block_idx=None, release=False):
        feat = []

        x = x.view(-1, self.nc, self.img_size, self.img_size).to(next(self.parameters()).device)
        x = self.block1(x)

        x = self.se_block1(x)
        x = self.triplet(x)
        feat.append(x)

        x_out = self.block2_mix(x)
        x_out = self.se_block2(x_out)
        x_out = self.cbam(x_out)
        feat.append(x_out)

        x = self.up(x_out)
        x = self.triplet_cloud(x)
        feat.append(x)

        x = self.block3(x)
        feat.append(x)

        x = self.block4(x)
        feat.append(x)

        x = x.contiguous().view(-1, self.ndf * 8 * int((self.img_size / 16)) * int((self.img_size / 16)))
        out = self.fc(x)

        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, nc=3, img_size=64):
        super(ResNet, self).__init__()
        self.inchannel = img_size
        self.mid = int(img_size / 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)

        # Add SE module after block2_mix
        self.se_block1 = SEModule(64, reduction=16)

        self.triplet = TripletAttention()

        # self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2_mix = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Add SE module after block2_mix
        self.se_block2 = SEModule(128, reduction=32)

        self.triplet_2 = TripletAttention()

        # self.layer2_x = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64)
        # )

        self.shortcut = nn.Sequential()

        self.block2_mix = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 6, 1, 1, 0),
            nn.Conv2d(6, 6, 3, 1, 1),
            nn.SiLU(True),
            nn.Conv2d(6, 6, 1, 1, 0),
            nn.Conv2d(6, 6, 3, 1, 1),
            nn.Conv2d(6, 6, 5, 1, 2),
            nn.ReLU(True),
        )

        # Add SE module after block2_mix
        self.se_block3 = SEModule(6, reduction=3)
        # self.se_block_without_attention = SEModule(64, reduction=16)

        self.cbam = CBAM(6, ratio=3)
        # self.cbam_without_attention = CBAM(64, ratio=32)
        # self.sa2 = SelfAttention(12, 16, num_head=2)

        # self.triplet_edge = TripletAttention()

        self.up = Up(6, 128)
        self.triplet_cloud = TripletAttention()

        # self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * (self.mid*2) * (self.mid*2), num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        cnt = 0
        for stride in strides:
            cnt += 1
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
            if self.inchannel == 64 and cnt == 2:
                self.inchannel = 128
        return nn.Sequential(*layers)

    def forward(self, x, feat_out=None, block_idx=None, release=False):
        feat = []
        x = self.conv1(x)
        out = self.layer1[0](x)
        # feat.append(out)
        for layer in self.layer1[1:]:
            out = layer(out)
            # feat.append(out)

        x = self.se_block1(x)
        x = self.triplet(x)
        feat.append(x)

        x_out = self.layer2_mix(x)
        x_out = self.se_block2(x_out)
        x_out = self.triplet_2(x_out)

        out = self.block2_mix(x_out)
        x_out = self.se_block3(out)
        out = self.cbam(x_out)
        # out = self.layer2_x(x_out)
        # out += self.shortcut(x_out)
        # x_out = F.relu(out)
        # x_out = self.se_block_without_attention(x_out)
        # out = self.cbam_without_attention(x_out)
        feat.append(out)

        # feat.append(out)

        out = self.up(out)
        out = self.triplet_cloud(out)
        feat.append(out)

        # out = self.layer2[0](out)
        # feat.append(out)
        # for layer in self.layer2[1:]:
        #     out = layer(out)
        #     feat.append(out)
        out = self.layer3[0](out)
        feat.append(out)
        for layer in self.layer3[1:]:
            out = layer(out)
            feat.append(out)
        out = self.layer4[0](out)
        feat.append(out)
        for layer in self.layer4[1:]:
            out = layer(out)
            feat.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat.append(out)
        out = self.fc(out)
        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


def rn18_pv4(num_classes, nc, img_size):
    return ResNet(ResidualBlock, num_classes, nc, img_size)


class VGG_pv4(nn.Module):

    def __init__(self, features, nc=3, num_class=10, img_size=64):
        super().__init__()
        self.features = features
        self.img_size = img_size
        self.mid = int(img_size / 64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.se_block1 = SEModule(64, reduction=16)
        self.triplet1 = TripletAttention()

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.se_block2 = SEModule(128, reduction=32)
        self.triplet2 = TripletAttention()

        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 4, 1, 1, 0),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.SiLU(True),
            nn.Conv2d(4, 4, 1, 1, 0),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.Conv2d(4, 4, 5, 1, 2),
            nn.ReLU(True),
        )

        self.se_block3 = SEModule(4, reduction=4)
        self.cbam = CBAM(4, ratio=4)

        self.up = Up(4, 256)
        # self.sa = SelfAttention(ndf * 2, 16)
        self.triplet_cloud = TripletAttention()

        self.block2_mix = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * (self.mid*2) * (self.mid*2), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_class)
        )

    def forward(self, x, block_idx=None, release=False):
        features = []
        out = self.conv1(x)
        x = self.se_block1(out)
        x = self.triplet1(x)
        features.append(x)

        out = self.conv2(x)
        x = self.se_block2(out)
        x = self.triplet2(x)
        features.append(x)

        x = self.layer1(x)
        x = self.se_block3(x)
        x = self.cbam(x)
        features.append(x)

        x = self.up(x)
        x = self.triplet_cloud(x)
        features.append(x)

        x = self.block2_mix(x)

        for i, layer in enumerate(self.features):
            x = layer(x)
            features.append(x)

        # x = self.avgpool(x)
        output = x.view(x.size()[0], -1)
        features.append(output)
        output = self.classifier(output)

        if release:
            return features, F.softmax(output, dim=1)
        else:
            return features, F.log_softmax(output, dim=1)


def make_layers(nc, cfg, batch_norm=False):
    layers = []

    input_channel = 256
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'D': [256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg16_bn_pv4(nc, nz, img_size):
    nc = nc
    nz = nz
    img_size = img_size
    return VGG_pv4(make_layers(nc, cfg['D'], batch_norm=True), nc, nz, img_size)