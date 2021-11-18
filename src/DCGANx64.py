import torch
from torch import nn

# -------------------- Discriminator --------------------


class Discriminator(nn.Module):
    """
    DCGAN Discriminator
    """

    def __init__(
        self,
        features_dim=64,
        img_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        """
        Discriminator Class

        Parameters
        ----------
        img_channels: int
            number if channels in image
        features_dim: int
            inner hidden dimension

        Returns
        ----------
        none
        """

        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # img_channels x 64 x 64
            nn.Conv2d(img_channels, features_dim, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
            # features_dim x 32 x 32
            nn.Conv2d(
                features_dim, features_dim * 2, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (features_dim * 2) x 16 x 16
            nn.Conv2d(
                features_dim * 2, features_dim * 4, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (features_dim * 4) x 8 x 8
            nn.Conv2d(
                features_dim * 4, features_dim * 8, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (features_dim * 8) x 4 x 4
            nn.Conv2d(features_dim * 8, 1, kernel_size, stride=1, padding=0),
            # 1 x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, image):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.

        Parameters
        ----------
        image: tensor
            a flattened image tensor with dimension (im_dim)

        Returns
        ----------
        _: tensor
            real or fake
        """
        return self.net(image)


# -------------------- Generator --------------------


class Generator(nn.Module):
    """
    DCGAN Generator
    """

    def __init__(
        self,
        z_dim,
        features_dim=64,
        img_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        """
        Generator Class

        Parameters
        ----------
        z_dim: int
            the dimension of the noise vector
        img_channels: int
            number if channels in image
        features_dim: int
            inner hidden dimension

        Returns
        ----------
        none
        """

        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # z_dim x 1 x 1
            nn.ConvTranspose2d(
                z_dim, features_dim * 16, kernel_size, stride=1, padding=0
            ),
            nn.BatchNorm2d(features_dim * 16),
            nn.ReLU(inplace=True),
            # (features_dim * 16) x 4 x 4
            nn.ConvTranspose2d(
                features_dim * 16,
                features_dim * 8,
                kernel_size,
                stride,
                padding,
            ),
            nn.BatchNorm2d(features_dim * 8),
            nn.ReLU(inplace=True),
            # (features_dim * 8) x 8 x 8
            nn.ConvTranspose2d(
                features_dim * 8, features_dim * 4, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 4),
            nn.ReLU(inplace=True),
            # (features_dim * 4) x 16 x 16
            nn.ConvTranspose2d(
                features_dim * 4, features_dim * 2, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(features_dim * 2),
            nn.ReLU(inplace=True),
            # (features_dim * 2) x 32 x 32
            nn.ConvTranspose2d(
                features_dim * 2, img_channels, kernel_size, stride, padding
            ),
            # img_channels x 64 x 64
            nn.Tanh(),
        )

    def forward(self, image):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.

        Parameters
        ----------
        noise: tensor
            a noise tensor with dimensions (n_samples, z_dim, 1, 1)

        Returns
        ----------
        _: image
            image after a forward pass
        """
        return self.net(image)
