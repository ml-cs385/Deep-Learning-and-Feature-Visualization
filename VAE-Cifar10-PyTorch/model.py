import torch
from torch import nn
from torch.autograd import Variable


class BN_VAE(nn.Module):
    def __init__(self, z_dim=512, kaiming=True):
        super(BN_VAE, self).__init__()

        self.z_dim = z_dim
        self.init_trick = kaiming

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, self.z_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.z_dim)
        self.fc21 = nn.Linear(self.z_dim, self.z_dim)
        self.fc22 = nn.Linear(self.z_dim, self.z_dim)

        # Decoder
        self.fc3 = nn.Linear(self.z_dim, self.z_dim)
        self.fc_bn3 = nn.BatchNorm1d(self.z_dim)
        self.fc4 = nn.Linear(self.z_dim, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        if self.init_trick:
            # add kaiming initialization trick
            nn.init.kaiming_normal_(self.conv1.weight)
            nn.init.kaiming_normal_(self.conv2.weight)
            nn.init.kaiming_normal_(self.conv3.weight)
            nn.init.kaiming_normal_(self.conv4.weight)
            nn.init.kaiming_normal_(self.conv5.weight)
            nn.init.kaiming_normal_(self.conv6.weight)
            nn.init.kaiming_normal_(self.conv7.weight)
            nn.init.kaiming_normal_(self.conv8.weight)

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        conv8 = self.conv8(conv7).view(-1, 3, 32, 32)
        return self.sigmoid(conv8)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class MAXPOOL_VAE(nn.Module):

    def __init__(self, z_dim=256):
        super(MAXPOOL_VAE, self).__init__()
        
        self.h_dim = 32*4*4
        self.z_dim = z_dim
        self.relu = nn.ReLU()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), #[bs, 16, 32, 32]
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16), #[bs, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #[bs, 32, 16, 16]
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32), #[bs, 32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), #[bs, 32, 8, 8]
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32), #[bs, 32, 4, 4]
            nn.ReLU()
        )
        self.fc0 = nn.Linear(self.h_dim, self.h_dim)
        self.fc0_bn = nn.BatchNorm1d(self.h_dim)

        self.fc1 = nn.Linear(self.h_dim, self.z_dim) # mu, [bs, 32*4*4=512] --> [bs, 512]
        self.fc2 = nn.Linear(self.h_dim, self.z_dim) # var, [bs, 32*4*4=512] --> [bs, 512]
        self.fc3 = nn.Linear(self.z_dim, self.h_dim) # [bs, 512] ---> [bs, 32*4*4]

        '''
           def _check_size_scale_factor(dim):
                if size is None and scale_factor is None:
                    raise ValueError('either size or scale_factor should be defined')
                if size is not None and scale_factor is not None:
                    raise ValueError('only one of size or scale_factor should be defined')
                if scale_factor is not None and isinstance(scale_factor, tuple)\
                        and len(scale_factor) != dim:
                    raise ValueError('scale_factor shape must match input shape. '
                                     'Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))
        '''

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Upsample(scale_factor=2), # [bs, 32, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Upsample(scale_factor=2), # [bs, 16, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Upsample(scale_factor=2), # [bs, 3, 32, 32]
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x).view(-1, self.h_dim)
        h = self.relu(self.fc0_bn(self.fc0(h)))
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z).view(-1, 32, 4, 4) # [bs, 32, 4, 4]
        res = self.decoder(z)
        return res

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

class LinearVAE(nn.Module):
    def __init__(self, z_dim=256, batch_size=128):
        super(LinearVAE, self).__init__()
        
        self.batch_size = batch_size
        self.h_dim = 32*4*4
        self.z_dim = z_dim
        self.relu = nn.ReLU()

        self.encoder = nn.Sequential(
            nn.Linear(3*32*32, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
        )
        self.fc0 = nn.Linear(self.h_dim, self.h_dim)
        self.fc0_bn = nn.BatchNorm1d(self.h_dim)

        self.fc1 = nn.Linear(self.h_dim, self.z_dim) # mu, [bs, 32*4*4=512] --> [bs, 512]
        self.fc2 = nn.Linear(self.h_dim, self.z_dim) # var, [bs, 32*4*4=512] --> [bs, 512]
        self.fc3 = nn.Linear(self.z_dim, self.h_dim) # [bs, 512] ---> [bs, 32*4*4]

        self.decoder = nn.Sequential(
            nn.Linear(self.h_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 3*32*32),
            nn.BatchNorm1d(3*32*32),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x): # shape of x: torch.Size([128, 3, 32, 32])
        h = self.encoder(x.view(-1, 3*32*32))
        h = self.relu(self.fc0_bn(self.fc0(h)))
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        res = self.decoder(z)
        return res

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


