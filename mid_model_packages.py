from __future__ import print_function

import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


class Classifier_4_mid(nn.Module):
    def __init__(self, nc, ndf, nz, img_size, block_idx=0):
        super(Classifier_4_mid, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.img_size = img_size

        self.block1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
        )

        self.k = ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)) // 2
        self.st_layer = nn.Linear(ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)), self.k * 2)

        self.fc = nn.Sequential(
            nn.Linear(self.k, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def _record_time_memory(self, block, x, device):
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            x = block(x)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # Milliseconds
            memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)  # Megabytes

        else:
            # CPU Time and Memory
            start_time = time.time()  # Record CPU start time
            memory_usage_before = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory before in MB

            x = block(x)

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            memory_usage_after = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory after in MB
            memory_usage = memory_usage_after - memory_usage_before  # Memory difference in MB

        return x, elapsed_time, memory_usage

    def forward(self, x, block_idx=None, record_time_memory=False, release=False, device="cuda"):
        times, feat, memories = [], [], []
        blocks = [self.block1, self.block2, self.block3, self.block4]

        x = x.view(-1, self.nc, self.img_size, self.img_size)

        for idx, block in enumerate(blocks):
            if record_time_memory:
                x, elapsed_time, memory_usage = self._record_time_memory(block, x, device)
                times.append(elapsed_time)
                memories.append(memory_usage)
            else:
                x = block(x)

            feat.append(x)

        # x = self.block1(x)
        # feat.append(x)
        # x = self.block2(x)
        # feat.append(x)
        # x = self.block3(x)
        # feat.append(x)
        # x = self.block4(x)
        # feat.append(x)
        x = x.view(-1, self.ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)))

        statis = self.st_layer(x)
        mu, std = statis[:,:self.k], statis[:,self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps

        out = self.fc(res)


        if release:
            return feat, mu, std, F.softmax(out, dim=1)
        else:
            return feat, mu, std, F.log_softmax(out, dim=1)


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
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        self.k = 512 * (self.mid*2) * (self.mid*2)
        self.st_layer = nn.Linear(512 * (self.mid*2) * (self.mid*2), self.k * 2)

        self.fc = nn.Sequential(
            nn.Linear(self.k, num_classes * 5),
            nn.Dropout(0.5),
            nn.Linear(num_classes * 5, num_classes),
        )

        # self.fc = nn.Linear(512 * (self.mid*2) * (self.mid*2), num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, block_idx=None, release=False):
        feat = []
        out = self.conv1(x)
        feat.append(out)
        out = self.layer1[0](out)
        feat.append(out)
        for layer in self.layer1[1:]:
            out = layer(out)
            feat.append(out)
        out = self.layer2[0](out)
        feat.append(out)
        for layer in self.layer2[1:]:
            out = layer(out)
            feat.append(out)
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

        x = out.view(out.size(0), -1)

        statis = self.st_layer(x)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps

        out = self.fc(res)

        if release:
            return feat, mu, std, F.softmax(out, dim=1)
        else:
            return feat, mu, std, F.log_softmax(out, dim=1)


def rn18_mid(num_classes, nc, img_size):
    return ResNet(ResidualBlock, num_classes, nc, img_size)