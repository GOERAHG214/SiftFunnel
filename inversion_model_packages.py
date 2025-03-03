from __future__ import print_function
import torch.nn as nn


class Inversion_4(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(Inversion_4, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        # Adjust the input channels and feature map size based on the selected block
        if block_idx == 4:
            input_channels = ndf * 8  # block4 output (ndf*8) x 4 x 4
            self.spatial_size = img_size // 16
            self.decoder = self._create_decoder(input_channels, 4)  # 4 transpose conv layers
        elif block_idx == 3:
            input_channels = ndf * 4  # block3 output (ndf*4) x 8 x 8
            self.spatial_size = img_size // 8
            self.decoder = self._create_decoder(input_channels, 3)  # 3 transpose conv layers
        elif block_idx == 2:
            input_channels = ndf * 2  # block2 output (ndf*2) x 16 x 16
            self.spatial_size = img_size // 4
            self.decoder = self._create_decoder(input_channels, 2)  # At least 2 transpose conv layers
        elif block_idx == 1:
            input_channels = ndf      # block1 output (ndf) x 32 x 32
            self.spatial_size = img_size // 2
            self.decoder = self._create_decoder(input_channels, 1)  # At least 1 transpose conv layer
        elif block_idx == 5:  # If it's from the fully connected layer
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),  # Adjusted according to your request
                nn.BatchNorm2d(ngf * 8),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()  # Final output size (nc) x 64 x 64
            )
        else:
            raise ValueError("block_idx must be between 1 and 4, or 'fc'")

    def _create_decoder(self, input_channels, num_layers):
        # Dynamically create the decoder based on the number of layers and the block selected
        layers = []
        ngf = input_channels // 2
        for i in range(num_layers, 0, -1):
            if self.spatial_size == self.img_size // 2:
                layers.append(nn.ConvTranspose2d(input_channels, self.nc, 4, 2, 1))
                layers.append(nn.Sigmoid())  # Output image size is (nc) x 64 x 64
                break
            else:
                layers.append(nn.ConvTranspose2d(input_channels, ngf, 4, 2, 1))
                layers.append(nn.BatchNorm2d(ngf))
                layers.append(nn.Tanh())  # ReLU replaced by Tanh
                input_channels = ngf
                ngf = ngf // 2
                self.spatial_size *= 2

        return nn.Sequential(*layers)

    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        if self.block_idx == 5:
            x = x.view(-1, self.nz, 1, 1)
            x = self.decoder(x)
            x = x.view(-1, self.nc, self.img_size, self.img_size)  # Reshape back to image size
        else:
            x = self.decoder(x)
        return x


class AE_Inversion_4(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(AE_Inversion_4, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        # Adjust the input channels and feature map size based on the selected block
        if block_idx == 4:
            input_channels = ndf * 8  # block4 output (ndf*8) x 4 x 4
            self.spatial_size = img_size // 16
            self.decoder = self._create_decoder(input_channels, 4)  # 4 transpose conv layers
        elif block_idx == 3:
            input_channels = ndf * 4  # block3 output (ndf*4) x 8 x 8
            self.spatial_size = img_size // 8
            self.decoder = self._create_decoder(input_channels, 3)  # 3 transpose conv layers
        elif block_idx == 2:
            # input_channels = ndf  # block2 output (ndf*2) x 16 x 16
            input_channels = ndf
            self.spatial_size = img_size // 4
            # self.decoder = nn.Sequential(
            #     nn.ConvTranspose2d(input_channels, ndf, 4, 2, 1),  # Adjusted according to your request
            #     nn.BatchNorm2d(ndf),
            #     nn.Tanh(),
            #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            #     nn.Sigmoid()  # Final output size (nc) x 64 x 64
            # )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(input_channels, ndf*2, 3, 1, 1),  # Adjusted according to your request
                nn.BatchNorm2d(ndf*2),
                nn.Tanh(),
                nn.ConvTranspose2d(ndf*2, ndf*2, 3, 1, 1),
                nn.ConvTranspose2d(ndf * 2, ndf * 2, 3, 1, 1),
                nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1),  # Adjusted according to your request
                nn.BatchNorm2d(ndf),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()  # Final output size (nc) x 64 x 64
            )
        elif block_idx == 1:
            input_channels = ndf      # block1 output (ndf) x 32 x 32
            self.spatial_size = img_size // 2
            self.decoder = self._create_decoder(input_channels, 1)  # At least 1 transpose conv layer
        elif block_idx == 5:  # If it's from the fully connected layer
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),  # Adjusted according to your request
                nn.BatchNorm2d(ngf * 8),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()  # Final output size (nc) x 64 x 64
            )
        else:
            raise ValueError("block_idx must be between 1 and 4, or 'fc'")

    def _create_decoder(self, input_channels, num_layers):
        # Dynamically create the decoder based on the number of layers and the block selected
        layers = []
        ngf = input_channels // 2
        # layers.append(nn.ConvTranspose2d(input_channels, input_channels, 4, 2, 1))
        # layers.append(nn.BatchNorm2d(input_channels))
        # layers.append(nn.Tanh())
        # layers.append(nn.ConvTranspose2d(input_channels, input_channels * 2, 3, 1, 1))
        # layers.append(nn.ConvTranspose2d(input_channels*2, input_channels * 2, 3, 1, 1))
        # layers.append(nn.Tanh())
        # layers.append(nn.ConvTranspose2d(input_channels * 2, input_channels * 2, 3, 1, 1))
        # layers.append(nn.ConvTranspose2d(input_channels * 2, input_channels * 2, 3, 1, 1))
        #
        # layers.append(nn.ConvTranspose2d(input_channels*2, input_channels, 4, 2, 1))
        # layers.append(nn.BatchNorm2d(input_channels))
        # layers.append(nn.Tanh())
        # self.spatial_size *= 2
        for i in range(num_layers, 0, -1):
            if self.spatial_size == self.img_size // 2:
                layers.append(nn.ConvTranspose2d(input_channels, self.nc, 4, 2, 1))
                layers.append(nn.Sigmoid())  # Output image size is (nc) x 64 x 64
                break
            else:
                layers.append(nn.ConvTranspose2d(input_channels, ngf, 4, 2, 1))
                layers.append(nn.BatchNorm2d(ngf))
                layers.append(nn.Tanh())  # ReLU replaced by Tanh
                input_channels = ngf
                ngf = ngf // 2
                self.spatial_size *= 2

        return nn.Sequential(*layers)

    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        if self.block_idx == 5:
            x = x.view(-1, self.nz, 1, 1)
            x = self.decoder(x)
            x = x.view(-1, self.nc, self.img_size, self.img_size)  # Reshape back to image size
        else:
            x = self.decoder(x)
        return x


class pv4_Inversion_4(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(pv4_Inversion_4, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        # Adjust the input channels and feature map size based on the selected block
        if block_idx == 4:
            input_channels = ndf * 8  # block4 output (ndf*8) x 4 x 4
            self.spatial_size = img_size // 16
            # self.decoder = self._create_decoder(input_channels, 4)  # 4 transpose conv layers
        elif block_idx == 3:
            input_channels = ndf * 2  # block3 output (ndf*4) x 8 x 8
            self.spatial_size = img_size // 8
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(input_channels, 256, 3, 1, 1),  # Adjusted according to your request
                nn.BatchNorm2d(256),
                nn.Tanh(),
                nn.ConvTranspose2d(256, 12, 3, 1, 1),  # Adjusted according to your request
                nn.BatchNorm2d(12),
                nn.Tanh(),
                nn.ConvTranspose2d(12, 12, 3, 1, 1),
                nn.ConvTranspose2d(12, 12, 3, 1, 1),
                nn.Tanh(),
                nn.ConvTranspose2d(12, 12, 3, 1, 1),
                nn.ConvTranspose2d(12, ndf // 4, 3, 1, 1),
                nn.ConvTranspose2d(ndf // 4, ndf, 4, 2, 1),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()  # Final output size (nc) x 64 x 64
            )
            # self.decoder = self._create_decoder(input_channels, 3)  # 3 transpose conv layers
        elif block_idx == 2:
            # input_channels = ndf  # block2 output (ndf*2) x 16 x 16
            input_channels = 12
            self.spatial_size = img_size // 4
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(input_channels, 12, 3, 1, 1),  # Adjusted according to your request
                nn.BatchNorm2d(12),
                nn.Tanh(),
                nn.ConvTranspose2d(input_channels, 12, 3, 1, 1),
                nn.ConvTranspose2d(input_channels, 12, 3, 1, 1),
                nn.Tanh(),
                nn.ConvTranspose2d(input_channels, 12, 3, 1, 1),
                nn.ConvTranspose2d(input_channels, ndf // 4, 3, 1, 1),
                nn.ConvTranspose2d(ndf // 4, ndf, 4, 2, 1),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()  # Final output size (nc) x 64 x 64
            )
            # self.decoder = nn.Sequential(
            #     nn.ConvTranspose2d(input_channels, ndf, 4, 2, 1),  # Adjusted according to your request
            #     nn.BatchNorm2d(ndf),
            #     nn.Tanh(),
            #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            #     nn.Sigmoid()  # Final output size (nc) x 64 x 64
            # )
        elif block_idx == 1:
            input_channels = ndf      # block1 output (ndf) x 32 x 32
            self.spatial_size = img_size // 2
            self.decoder = self._create_decoder(input_channels, 1)  # At least 1 transpose conv layer
        elif block_idx == 5:  # If it's from the fully connected layer
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),  # Adjusted according to your request
                nn.BatchNorm2d(ngf * 8),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()  # Final output size (nc) x 64 x 64
            )
        else:
            raise ValueError("block_idx must be between 1 and 4, or 'fc'")

    def _create_decoder(self, input_channels, num_layers):
        # Dynamically create the decoder based on the number of layers and the block selected
        layers = []
        ngf = input_channels // 2
        # layers.append(nn.ConvTranspose2d(input_channels, input_channels, 4, 2, 1))
        # layers.append(nn.BatchNorm2d(input_channels))
        # layers.append(nn.Tanh())
        # layers.append(nn.ConvTranspose2d(input_channels, input_channels * 2, 3, 1, 1))
        # layers.append(nn.ConvTranspose2d(input_channels*2, input_channels * 2, 3, 1, 1))
        # layers.append(nn.Tanh())
        # layers.append(nn.ConvTranspose2d(input_channels * 2, input_channels * 2, 3, 1, 1))
        # layers.append(nn.ConvTranspose2d(input_channels * 2, input_channels * 2, 3, 1, 1))
        #
        # layers.append(nn.ConvTranspose2d(input_channels*2, input_channels, 4, 2, 1))
        # layers.append(nn.BatchNorm2d(input_channels))
        # layers.append(nn.Tanh())
        # self.spatial_size *= 2
        for i in range(num_layers, 0, -1):
            if self.spatial_size == self.img_size // 2:
                layers.append(nn.ConvTranspose2d(input_channels, self.nc, 4, 2, 1))
                layers.append(nn.Sigmoid())  # Output image size is (nc) x 64 x 64
                break
            else:
                layers.append(nn.ConvTranspose2d(input_channels, ngf, 4, 2, 1))
                layers.append(nn.BatchNorm2d(ngf))
                layers.append(nn.Tanh())  # ReLU replaced by Tanh
                input_channels = ngf
                ngf = ngf // 2
                self.spatial_size *= 2

        return nn.Sequential(*layers)

    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        if self.block_idx == 5:
            x = x.view(-1, self.nz, 1, 1)
            x = self.decoder(x)
            x = x.view(-1, self.nc, self.img_size, self.img_size)  # Reshape back to image size
        else:
            x = self.decoder(x)
        return x


class InversionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(InversionResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class InversionResNet(nn.Module):
    def __init__(self, InversionResidualBlock, block_idx, num_classes=10, nc=3, img_size=64):
        super(InversionResNet, self).__init__()
        self.img_size = img_size
        self.block_idx = block_idx
        self.ndf = 64  # Number of feature maps in the first convolutional layer

        self.deconv_layers = []

        # Define residual blocks based on block_idx
        if block_idx <= 3:
            # Block 0-14
            if block_idx == 3 or 2:
                self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(4, 6):
            # Block 15-28
            if block_idx == 5:
                self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(6, 8):
            # Block 29-42
            if block_idx == 7:
                self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(8, 11):
            # Block 43-57
            if block_idx == 9 or 10:
                self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 256, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        else:
            raise ValueError("block_idx out of range")

        # Convert list to Sequential
        self.deconv_layers = nn.Sequential(*self.deconv_layers)

        # Define final layers to get to the original image size
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(final_channels, nc, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layers(x)
        return x


def inversion_resnet(block_idx, num_classes, nc, img_size):
    return InversionResNet(InversionResidualBlock, block_idx, num_classes, nc, img_size)


class InversionResidualBlock_pv4(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(InversionResidualBlock_pv4, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class InversionResNet_pv4(nn.Module):
    def __init__(self, InversionResidualBlock, block_idx, num_classes=10, nc=3, img_size=64):
        super(InversionResNet_pv4, self).__init__()
        self.img_size = img_size
        self.block_idx = block_idx
        self.ndf = 64  # Number of feature maps in the first convolutional layer

        self.deconv_layers = []

        self.deconv_layers1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1,
                               output_padding=0,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(64),
        )

        self.deconv_layers2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1,
                               output_padding=0,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(64),
        )

        # Define residual blocks based on block_idx
        if block_idx <= 3:
            # Block 0-14
            if block_idx == 3:
                self.decoder1 = nn.Sequential(
                    nn.ConvTranspose2d(64, 2, 3, 1, 1),  # Adjusted according to your request
                    nn.BatchNorm2d(2),
                    nn.Tanh(),
                )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(2, 16, 3, 1, 1),  # Adjusted according to your request
                nn.BatchNorm2d(16),
                nn.Tanh(),
                nn.ConvTranspose2d(16, 16, 3, 1, 1),
                nn.ConvTranspose2d(16, 16, 3, 1, 1),
                nn.Tanh(),
                nn.ConvTranspose2d(16, 16, 3, 1, 1),
                nn.ConvTranspose2d(16, 64, 3, 1, 1),
            )
            # self.deconv_layers.append(InversionResidualBlock(2, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))

            final_channels = 64
        elif block_idx in range(4, 6):
            # Block 15-28
            if block_idx == 5:
                self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(6, 8):
            # Block 29-42
            if block_idx == 7:
                self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(8, 11):
            # Block 43-57
            if block_idx == 9 or 10:
                self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 256, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        else:
            raise ValueError("block_idx out of range")

        # Convert list to Sequential
        self.deconv_layers = nn.Sequential(*self.deconv_layers)

        # Define final layers to get to the original image size
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(final_channels, nc, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.deconv_layers1(x)
        # x = self.deconv_layers2(x)
        x = self.decoder(x)
        x = self.deconv_layers(x)
        x = self.final_layers(x)
        return x


def inversion_resnet_pv4(block_idx, num_classes, nc, img_size):
    return InversionResNet_pv4(InversionResidualBlock_pv4, block_idx, num_classes, nc, img_size)


class InversionResNet_AE(nn.Module):
    def __init__(self, InversionResidualBlock, block_idx, num_classes=10, nc=3, img_size=64):
        super(InversionResNet_AE, self).__init__()
        self.img_size = img_size
        self.block_idx = block_idx
        self.ndf = 64  # Number of feature maps in the first convolutional layer

        self.deconv_layers = []

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, 1, 1),  # Adjusted according to your request
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),  # Adjusted according to your request
        )

        # Define residual blocks based on block_idx
        if block_idx <= 3:
            # Block 0-14
            if block_idx == 3:
                self.deconv_layers1 = nn.Sequential(
                    nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1,
                                       output_padding=0,
                                       bias=False),
                    nn.BatchNorm2d(64),
                    nn.Tanh(),
                    nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0,
                                       bias=False),
                    nn.BatchNorm2d(64),
                )
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(4, 6):
            # Block 15-28
            if block_idx == 5:
                self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(6, 8):
            # Block 29-42
            if block_idx == 7:
                self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(8, 11):
            # Block 43-57
            if block_idx == 9 or 10:
                self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 256, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        else:
            raise ValueError("block_idx out of range")

        # Convert list to Sequential
        self.deconv_layers = nn.Sequential(*self.deconv_layers)

        # Define final layers to get to the original image size
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(final_channels, nc, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = self.deconv_layers1(x)
        x = self.deconv_layers(x)
        x = self.final_layers(x)
        return x


def inversion_resnet_AE(block_idx, num_classes, nc, img_size):
    return InversionResNet_AE(InversionResidualBlock, block_idx, num_classes, nc, img_size)


class inversion_vgg(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(inversion_vgg, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, nc, 3, 1, 1),
            nn.Sigmoid()  # Final output size (nc) x 64 x 64
        )


    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        x = self.decoder(x)
        return x


class inversion_vgg_pv4(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(inversion_vgg_pv4, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        self.decoderx = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 3, 1, 1),  # Adjusted according to your request
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.ConvTranspose2d(4, 4, 3, 1, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(4, 32, 3, 1, 1),
            nn.ConvTranspose2d(32, 128, 4, 2, 1),  # Adjusted according to your request
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(8, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, nc, 3, 1, 1),
            nn.Sigmoid()  # Final output size (nc) x 64 x 64
        )


    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        x = self.decoderx(x)
        x = self.decoder(x)
        return x