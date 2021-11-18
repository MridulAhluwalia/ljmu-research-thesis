import torch
from torch import nn

# -------------------- Generator --------------------


class Generator(nn.Module):
    def __init__(self, z_dim, im_dim, hidden_dim):
        """
        Generator Class

        Parameters
        ----------
        z_dim: int
            the dimension of the noise vector
        im_dim: int
            the dimension of the images, length x width
        hidden_dim: int
            inner hidden dimension

        Returns
        ----------
        none
        """

        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self.generator_block(z_dim, hidden_dim),
            self.generator_block(hidden_dim, hidden_dim * 2),
            self.generator_block(hidden_dim * 2, hidden_dim * 4),
            self.generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.

        Parameters
        ----------
        noise: tensor
            a noise tensor with dimensions (n_samples, z_dim)

        Returns
        ----------
        _: image
            image after a forward pass
        """
        return self.gen(noise)

    def generator_block(self, input_dim, output_dim):
        """
        Function for returning a block of the generator's neural network
        given input and output dimensions.

        Parameters
        ----------
            input_dim: int
                the dimension of the input vector
            output_dim: int
                the dimension of the output vector

        Returns
        ----------
            a generator neural network layer, with a linear transformation
            followed by a batch normalization and then a relu activation
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def cal_generator_loss(
        self, disc, criterion, real, num_images, z_dim, device
    ):
        """
        Return the loss of the discriminator given inputs

        Parameters
        ----------
        gen: the generator model
            which returns an image given z-dimensional noise
        disc: the discriminator model
            which returns a single-dimensional prediction of real/fake
        criterion: the loss function
            which should be used to compare
            the discriminator's predictions to the ground truth reality of the images
            (e.g. fake = 0, real = 1)
        real: _
            a batch of real images
        num_images: int
            the number of images the generator should produce,
            which is also the length of the real images
        z_dim: int
            the dimension of the noise vector
        device: str
            the device type

        Returns
        ----------
        gen_loss: int
            a torch scalar loss value for the current batch
        """

        fake_noise = get_noise(num_images, z_dim, device=device)
        fake = self.gen(fake_noise)
        disc_fake_pred = disc(fake)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss


# -------------------- Discriminator --------------------


class Discriminator(nn.Module):
    def __init__(self, im_dim, hidden_dim):
        """
        Discriminator Class

        Parameters
        ----------
        im_dim: int
            the dimension of the images, length x width
        hidden_dim: int
            inner hidden dimension

        Returns
        ----------
        none
        """

        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            self.discriminator_block(im_dim, hidden_dim * 4),
            self.discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
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
        return self.disc(image)

    def discriminator_block(self, input_dim, output_dim):
        """
        Function for returning a neural network of the discriminator given input and output dimensions.

        Parameters
        ----------
        input_dim: int
            the dimension of the input vector
        output_dim: int
            the dimension of the output vector

        Returns
        ----------
            a discriminator neural network layer, with a linear transformation
            followed by an nn.LeakyReLU activation with negative slope of 0.2
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),  # Layer 1
            nn.LeakyReLU(0.2, inplace=True),
        )

    def cal_discriminator_loss(
        self, gen, criterion, real, num_images, z_dim, device
    ):
        """
        Return the loss of the discriminator given inputs

        Parameters
        ----------
        gen: the generator model
            which returns an image given z-dimensional noise
        disc: the discriminator model
            which returns a single-dimensional prediction of real/fake
        criterion: the loss function
            which should be used to compare
            the discriminator's predictions to the ground truth reality of the images
            (e.g. fake = 0, real = 1)
        real: _
            a batch of real images
        num_images: int
            the number of images the generator should produce,
            which is also the length of the real images
        z_dim: int
            the dimension of the noise vector
        device: str
            the device type

        Returns
        ----------
        disc_loss: int
            a torch scalar loss value for the current batch
        """

        fake_noise = get_noise(num_images, z_dim, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = self.disc(fake.detach())
        disc_fake_loss = criterion(
            disc_fake_pred, torch.zeros_like(disc_fake_pred)
        )
        disc_real_pred = self.disc(real)
        disc_real_loss = criterion(
            disc_real_pred, torch.ones_like(disc_real_pred)
        )
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss
