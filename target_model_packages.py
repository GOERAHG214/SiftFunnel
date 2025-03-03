from __future__ import print_function

import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


class Classifier_4(nn.Module):
    def __init__(self, nc, ndf, nz, img_size, block_idx=0):
        super(Classifier_4, self).__init__()

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

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)), nz * 5),
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
        # out = self.fc(x)

        if record_time_memory:
            out, elapsed_time, memory_usage = self._record_time_memory(self.fc, x, device)
            times.append(elapsed_time)
            memories.append(memory_usage)
        else:
            out = self.fc(x)

        if record_time_memory:
            return feat, times, memories
        else:
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
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * (self.mid*2) * (self.mid*2), num_classes)

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
        out = out.view(out.size(0), -1)
        feat.append(out)
        out = self.fc(out)
        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


def rn18(num_classes, nc, img_size):
    return ResNet(ResidualBlock, num_classes, nc, img_size)


class VGG(nn.Module):

    def __init__(self, features, num_class=10, img_size=64):
        super().__init__()
        self.features = features
        self.img_size = img_size
        self.mid = int(img_size / 64)

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
        for i, layer in enumerate(self.features):
            x = layer(x)
            features.append(x)

        # x = self.avgpool(x)
        output = x.view(x.size()[0], -1)
        output = self.classifier(output)

        if release:
            return features, F.softmax(output, dim=1)
        else:
            return features, F.log_softmax(output, dim=1)


def make_layers(nc, cfg, batch_norm=False):
    layers = []

    input_channel = nc
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
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg16_bn(nc, nz, img_size):
    nc = nc
    nz = nz
    img_size = img_size
    return VGG(make_layers(nc, cfg['D'], batch_norm=True), nz, img_size)


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Backbone64(Module):
    def __init__(self, input_size, num_layers, nc, mode='ir'):
        super(Backbone64, self).__init__()
        assert input_size[0] in [64, 128], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(),
                                       Flatten(),
                                       Linear(512 * 14 * 14, 10),
                                       BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        features = []
        x = self.input_layer(x)
        features.append(x)
        for layer in self.body:
            x = layer(x)
            features.append(x)
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class IR152(nn.Module):
    def __init__(self, num_classes=10, nc=3, img_size=64):
        super(IR152, self).__init__()
        self.feature = IR_152_64((img_size, img_size), nc)
        self.feat_dim = 512
        self.num_classes = num_classes
        self.mid = int(img_size / 16)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * self.mid * self.mid, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x, release=False):
        feat = self.feature(x)
        feat = feat[-1]
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


def IR_152_64(input_size, nc):
    """Constructs a ir-152 model.
    """
    model = Backbone64(input_size, 152, nc, 'ir')

    return model


class ResNetWithProfiler(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, nc=3, img_size=64):
        super(ResNetWithProfiler, self).__init__()
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
        self.fc = nn.Linear(512 * (self.mid * 2) * (self.mid * 2), num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, release=False):
        feat = []
        out = self.conv1(x)
        feat.append(out)
        out = self.layer1[0](out)
        feat.append(out)
        for layer in self.layer1[1:]:
            out = layer(out)
            feat.append(out)
        # out = self.layer2[0](out)
        # feat.append(out)
        # for layer in self.layer2[1:]:
        #     out = layer(out)
        #     feat.append(out)
        # out = self.layer3[0](out)
        # feat.append(out)
        # for layer in self.layer3[1:]:
        #     out = layer(out)
        #     feat.append(out)
        # out = self.layer4[0](out)
        # feat.append(out)
        # for layer in self.layer4[1:]:
        #     out = layer(out)
        #     feat.append(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # feat.append(out)
        # out = self.fc(out)
        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)

    def _record_time_memory(self, input_tensor, block_idx, device):
        self.to(device)
        input_tensor = input_tensor.to(device)

        if device == 'cuda':
            torch.cuda.synchronize()  # Synchronize before timing
            start_time = time.time()

            with torch.no_grad():
                feat, _ = self(input_tensor, release=True)  # Perform forward pass
                torch.cuda.synchronize()  # Wait for all CUDA operations to finish

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Record GPU memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
            memory_cached = torch.cuda.memory_reserved() / (1024 ** 2)  # in MB
            print(f'GPU - Block index {block_idx}:')
            print(f'Time taken: {elapsed_time:.4f} seconds')
            print(f'Memory allocated: {memory_allocated:.4f} MB')
            print(f'Memory cached: {memory_cached:.4f} MB')

        else:
            start_time = time.time()

            with torch.no_grad():
                feat, _ = self(input_tensor, release=True)  # Perform forward pass

            end_time = time.time()
            elapsed_time = end_time - start_time

            # For CPU, memory usage can be approximated
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            print(f'CPU - Block index {block_idx}:')
            print(f'Time taken: {elapsed_time:.4f} seconds')
            print(f'Memory allocated: {memory_allocated:.4f} MB')

        return elapsed_time, memory_allocated

    def profile(self, input_size=(1, 3, 64, 64), block_indices=None, device="cuda"):
        if block_indices is not None:
            block_indices = list(range(0, block_indices))  # Assuming block indices from 1 to 58

        input_tensor = torch.randn(*input_size)

        print(f'\nProfiling for block_idx: {block_indices}')
        self._record_time_memory(input_tensor, block_indices, device)

def rn18_test(num_classes, nc, img_size):
    return ResNetWithProfiler(ResidualBlock, num_classes, nc, img_size)



class DP_Drop_Classifier_4(nn.Module):
    def __init__(self, nc, ndf, nz, img_size):
        super(DP_Drop_Classifier_4, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.img_size = img_size

        self.block1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )
        self.block4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)), nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, feat_out=None, block_idx=None, release=False):
        feat = []

        x = x.view(-1, self.nc, self.img_size, self.img_size)
        x = self.block1(x)
        feat.append(x)
        if block_idx == 1:
            x = feat_out
        x = self.block2(x)
        feat.append(x)
        if block_idx == 2:
            x = feat_out
        x = self.block3(x)
        feat.append(x)
        if block_idx == 3:
            x = feat_out
        x = self.block4(x)
        feat.append(x)
        if block_idx == 4:
            x = feat_out
        x = x.view(-1, self.ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)))
        out = self.fc(x)

        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


class DP_Drop_ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, nc=3, img_size=64):
        super(DP_Drop_ResNet, self).__init__()
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
        self.fc = nn.Linear(512 * (self.mid * 2) * (self.mid * 2), num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, feat_out=None, block_idx=None, release=False):
        feat = []
        out = self.conv1(x)
        feat.append(out)

        if block_idx == 1:
            out = feat_out
        for idx, layer in enumerate(self.layer1):
            out = layer(out)
            feat.append(out)
            if block_idx == 2 + idx:
                out = feat_out

        for idx, layer in enumerate(self.layer2):
            out = layer(out)
            feat.append(out)
            if block_idx == 4 + idx:
                out = feat_out

        for idx, layer in enumerate(self.layer3):
            out = layer(out)
            feat.append(out)
            if block_idx == 6 + idx:
                out = feat_out

        for idx, layer in enumerate(self.layer4):
            out = layer(out)
            feat.append(out)
            if block_idx == 8 + idx:
                out = feat_out

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat.append(out)
        out = self.fc(out)

        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


def dp_drop_resnet18(num_classes, nc, img_size):
    return DP_Drop_ResNet(ResidualBlock, num_classes, nc, img_size)


class EncoderWithResidualEdge(nn.Module):
    def __init__(self, ndf):
        super(EncoderWithResidualEdge, self).__init__()
        self.conv2ed = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1),
        )
        self.res_edge = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf)
        )


    def forward(self, x):
        first_conv_out = self.conv2ed(x)

        encoder_out = self.encoder(x)

        combined_out = first_conv_out + encoder_out

        final_out = F.relu(self.res_edge(combined_out))

        return final_out


class DecoderWithResidualEdge(nn.Module):
    def __init__(self, ndf):
        super(DecoderWithResidualEdge, self).__init__()
        self.conv2ed = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(ndf, ndf, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(ndf, ndf, 3, 1, 1),
        )
        self.res_cloud = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
        )

    def forward(self, x):
        first_conv_out = self.conv2ed(x)

        encoder_out = self.encoder(x)

        combined_out = first_conv_out + encoder_out

        final_out = F.relu(self.res_cloud(combined_out))

        return final_out


class AE_Classifier_4(nn.Module):
    def __init__(self, nc, ndf, nz, img_size, block_idx=0, quantization_factor=16):
        super(AE_Classifier_4, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.img_size = img_size
        self.quantization_factor = quantization_factor

        # Define blocks and layers (same as before)
        self.block1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )
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

        self.encoder = EncoderWithResidualEdge(ndf)

        self.decoder = DecoderWithResidualEdge(ndf)

    def forward(self, x, feat_out=None, block_idx=None, release=False):
        feat = []

        x = x.view(-1, self.nc, self.img_size, self.img_size).to(next(self.parameters()).device)
        x = self.block1(x)
        feat.append(x)
        if block_idx == 1:
            x = feat_out
        x = self.block2(x)

        # Encoder
        x = self.encoder(x)

        feat.append(x)

        # Decoder
        x = self.decoder(x)
        feat.append(x)

        x = self.block3(x)
        feat.append(x)
        if block_idx == 3:
            x = feat_out
        x = self.block4(x)
        feat.append(x)
        if block_idx == 4:
            x = feat_out
        x = x.view(-1, self.ndf * 8 * int((self.img_size / 16)) * int((self.img_size / 16)))
        out = self.fc(x)

        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)



class ResNet_AE(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, nc=3, img_size=64):
        super(ResNet_AE, self).__init__()
        self.inchannel = img_size
        self.mid = int(img_size / 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)

        self.encoder = EncoderWithResidualEdge(32)

        self.decoder = DecoderWithResidualEdge(32)

        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * (self.mid*2) * (self.mid*2), num_classes)

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
        out = self.layer1[0](out)
        feat.append(out)
        for layer in self.layer1[1:]:
            out = layer(out)
            feat.append(out)

        out = self.encoder(out)
        feat.append(out)

        out = self.decoder(out)

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
        out = out.view(out.size(0), -1)
        feat.append(out)
        out = self.fc(out)
        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


def rn18_AE(num_classes, nc, img_size):
    return ResNet_AE(ResidualBlock, num_classes, nc, img_size)