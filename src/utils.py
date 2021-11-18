import torch

# -------------------- Noise --------------------


def get_noise(n_samples, z_dim, device="cpu"):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.

    Parameters
    ----------
    n_samples: int
        the number of samples to generate
    z_dim: int
        the dimension of the noise vector
    device: str
        the device type

    Returns
    ----------
    _: tensor
        tensor of that shape filled with random numbers from the normal distribution
    """
    return torch.randn(n_samples, z_dim, 1, 1, device=device)


# -------------------- Gradient Penalty --------------------


def get_gradient(crit, real, fake, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images

    Parameters
    ----------
    crit: model
        the critic model
    real: image tensor
        a batch of real images
    fake: image tensor
        a batch of fake images
    epsilon: float
        a vector of the uniformly random proportions of real/fake per mixed image

    Returns
    ----------
    gradient: float
        the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)
    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    """
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.

    Parameters
    ----------
    gradient: float
        the gradient of the critic's scores, with respect to the mixed image

    Returns
    ----------
    penalty: float
        the gradient penalty
    """
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)
    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


# -------------------- Loss Functions --------------------


def get_gen_loss(crit_fake_pred):
    """
    Return the loss of a generator given the critic's scores of the generator's fake images.

    Parameters
    ----------
    crit_fake_pred:
        the critic's scores of the fake images

    Returns
    ----------
    gen_loss: float
        a scalar loss value for the current batch of the generator
    """
    gen_loss = -1.0 * torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    """
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.

    Parameters
    ----------
    crit_fake_pred:
        the critic's scores of the fake images
    crit_real_pred:
        the critic's scores of the real images
    gp: float
        the unweighted gradient penalty
    c_lambda: float
        the current weight of the gradient penalty

    Returns
    ----------
    crit_loss: float
        a scalar for the critic's loss, accounting for the relevant factors
    """
    crit_loss = (
        torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    )
    return crit_loss
