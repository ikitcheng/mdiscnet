#@title VAE Class (by Sunwoong Yang)

# https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
import torch
from torch import nn

class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):

    def forward(self, input, size=64 * 1 * 1):

        return input.view(input.size(0), -1, 1, 1)


class DoubleConv(nn.Module):

    """(convolution => [BN] => ReLU) * 2"""


    def __init__(self, in_channels, out_channels, mid_channels=None, kernel = 3, padding = 1):

        super().__init__()

        if not mid_channels:

            mid_channels = out_channels

        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=padding, bias=False),

            nn.BatchNorm2d(mid_channels),

            nn.LeakyReLU()

            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),

            # nn.BatchNorm2d(out_channels),

            # nn.ReLU(inplace=True)

        )


    def forward(self, x):

        return self.double_conv(x)




class Down(nn.Module):

    """Downscaling with maxpool then double conv"""


    def __init__(self, in_channels, out_channels, scale=2):

        super().__init__()

        self.maxpool_conv = nn.Sequential(

            nn.MaxPool2d(scale),

            DoubleConv(in_channels, out_channels)

        )


    def forward(self, x):

        return self.maxpool_conv(x)




class Up(nn.Module):

    """Upscaling then double conv"""


    def __init__(self, in_channels, out_channels, scale=2, bilinear=True):

        super().__init__()


        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1):

        x1 = self.up(x1)

        return self.conv(x1)

        


class VAE(nn.Module):

    def __init__(self, image_channels=3, h_dim=64 * 1 * 1, z_dim=12, beta = 1, device=None):

    # 256 * 30 * 6

        super(VAE, self).__init__()

        self.beta = beta
        
        self.device = device

        self.encoder = nn.Sequential(
            # Input -1 x 512 x 128 (flow field data); -1 x 32 x 32 (MNIST images)

            DoubleConv(in_channels=image_channels, out_channels=64, kernel = 1), # 64 x 512 x 128; 64 x 32 x 32

            Down(64,64, scale=2), # 64 x 256 x 64; 64 x 16 x 16

            Down(64,64, scale=2), # 64 x 128 x 32; 64 x 8 x 8

            Down(64,64, scale=2), # 64 x 64 x 16; 64 x 4 x 4

            Down(64,64, scale=2), # 64 x 32 x 8; 64 x 2 x 2
            
            Down(64,64, scale=2), # 64 x 16 x 4; 64 x 1 x 1

#             Down(64,64, scale=2), # 64 x 8 x 2

#             Down(64,64, scale=2), # 64 x 4 x 1

            Flatten() # 256; 64

        )

        # h_dim = 256 * 30 * 6

        self.fc1 = nn.Linear(h_dim, z_dim) # z_dim 16; 8

        self.fc2 = nn.Linear(h_dim, z_dim) # z_dim 16; 8

        self.fc3 = nn.Linear(z_dim, h_dim) # just before decoder input sampled latent vector (z_dim 16; 8) -> FC3 -> hidden dimension 256; 64

        # self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim),nn.LeakyReLU())

        # self.fc2 = nn.Sequential(nn.Linear(h_dim, z_dim),nn.LeakyReLU())

        # self.fc3 = nn.Sequential(nn.Linear(z_dim, h_dim),nn.LeakyReLU()) # 디코더 입력 직전

        

        self.decoder = nn.Sequential(

            UnFlatten(), # 64 x 4 x 1; 64 x 1 x 1

            Up(64, 64, scale=2), # 64 x 8 x 2; 64 x 2 x 2

            Up(64, 64, scale=2), # 64 x 16 x 4; 64 x 4 x 4

            Up(64, 64, scale=2), # 64 x 32 x 8; 64 x 8 x 8

            Up(64, 64, scale=2), # 64 x 64 x 16; 64 x 16 x 16

            Up(64, 64, scale=2), # 64 x 128 x 32; 64 x 32 x 32

#             Up(64, 64, scale=2), # 64 x 256 x 64

#             Up(64, 64, scale=2), # 64 x 512 x 128

            # DoubleConv(64, 64),

            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, image_channels, kernel_size=1, stride=1, padding=0, bias=False), # back to the image dimensions -1 x 512 x 128 (flow field data); -1 x 32 x 32 (MNIST images)


            # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64, 126, 30

            # # nn.Upsample(scale_factor=2), # 132, 36

            # # nn.ReLU(),

            # # nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1), # 32, 255, 63

            # nn.Upsample(scale_factor=2),

            # nn.ReLU(),

            # nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1), # 3, 512, 128

            # nn.Sigmoid(),


        )

        

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_().to(self.device)

        if self.beta == 0:

          z = mu

        else:

          eps = torch.randn(*mu.size()).to(self.device)

          z = mu + std * eps

        return z

    

    def bottleneck(self, h):

        mu, logvar = self.fc1(h), self.fc2(h)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


    def encode(self, x):

        h = self.encoder(x)

        z, mu, logvar = self.bottleneck(h)

        return z, mu, logvar


    def decode(self, z):

        z = self.fc3(z)

        z = self.decoder(z)
        
        return z


    def forward(self, x):

        z, mu, logvar = self.encode(x)

        z = self.decode(z) # reconstruction -> recon_x

        return z, mu, logvar


    def encode_novar(self, x):

        h = self.encoder(x)

        mu, logvar = self.fc1(h), self.fc2(h)

        return mu


    def recon_novar(self, x):

        z = self.encode_novar(x)

        z = self.decode(z)

        return z