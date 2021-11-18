import logging
import os

import PIL.Image as Image

import ignite
import ignite.distributed as idist
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from DCGAN import Discriminator, Generator, get_noise
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import FID, SSIM, InceptionScore, RunningAverage
# from torchsummary import summary
from torchvision.datasets import ImageFolder

# -------------------- Initialization --------------------

torch.manual_seed(999)
ignite.utils.manual_seed(999)

ignite.utils.setup_logger(
    name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING
)
ignite.utils.setup_logger(
    name="ignite.distributed.launcher.Parallel", level=logging.WARNING
)

# -------------------- Configuration --------------------

image_size = 64
batch_size = 128
latent_dim = 100
n_channels = 3
n_epoch = 50
model = "DCGAN1"

input_data_path = "../data/processed/N"

data_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=idist.device())

# -------------------- Load Dataset --------------------

train_dataset = ImageFolder(root=input_data_path, transform=data_transform)
test_dataset = torch.utils.data.Subset(train_dataset, torch.arange(2048))

train_dataloader = idist.auto_dataloader(
    train_dataset,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,
)

test_dataloader = idist.auto_dataloader(
    test_dataset,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False,
)

# -------------------- Plot Data --------------------

real_batch = next(iter(train_dataloader))

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),
        (1, 2, 0),
    )
)
plt.show()

# -------------------- Load Models --------------------

netG = idist.auto_model(Generator(latent_dim))
summary(netG, (latent_dim, 1, 1))

netD = idist.auto_model(Discriminator())
summary(netD, (n_channels, image_size, image_size))

criterion = nn.BCELoss()

optimizerD = idist.auto_optim(
    optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
)

optimizerG = idist.auto_optim(
    optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
)

# -------------------- Training Setup --------------------


def training_step(engine, batch):
    # Set the models for training
    netG.train()
    netD.train()

    real, _ = batch

    real = real.idist.device()
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=idist.device())
    fake = netG(noise)

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################

    optimizerD.zero_grad()

    disc_fake_pred = netD(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

    disc_real_pred = netD(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    # Update gradients
    disc_loss.backward(retain_graph=True)
    # Update optimizer
    optimizerD.step()

    D_losses.append(disc_loss.item())

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################

    optimizerG.zero_grad()

    noise = torch.randn(batch_size, latent_dim, 1, 1, device=idist.device())
    fake = netG(noise)
    disc_fake_pred_2 = netD(fake)

    gen_loss = criterion(disc_fake_pred_2, torch.ones_like(disc_fake_pred_2))

    # Update gradients
    gen_loss.backward()
    # Update optimizer
    optimizerG.step()

    G_losses.append(gen_loss.item())

    return {
        "Loss_G": gen_loss.item(),
        "Loss_D": disc_loss.item(),
        "D_x": disc_real_pred.mean().item(),
        "D_G_z1": disc_fake_pred.mean().item(),
        "D_G_z2": disc_fake_pred_2.mean().item(),
    }


trainer = Engine(training_step)


def initialize_fn(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@trainer.on(Events.STARTED)
def init_weights():
    netD.apply(initialize_fn)
    netG.apply(initialize_fn)


G_losses = []
D_losses = []


@trainer.on(Events.ITERATION_COMPLETED)
def store_losses(engine):
    o = engine.state.output
    G_losses.append(o["Loss_G"])
    D_losses.append(o["Loss_D"])


img_list = []


@trainer.on(Events.EPOCH_COMPLETED)
def store_images(engine):
    with torch.no_grad():
        fake = netG(fixed_noise).cpu()
    img_list.append(fake)

    plt.title("Fake Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(img_list[-1], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()


# -------------------- Evaluation Setup --------------------

fid_metric = FID(device=idist.device())
is_metric = InceptionScore(
    device=idist.device(), output_transform=lambda x: x[0]
)
ssim_metric = SSIM(data_range=1.0)


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def evaluation_step(engine, batch):
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=idist.device())
        netG.eval()
        fake_batch = netG(noise)
        fake = interpolate(fake_batch)
        real = interpolate(batch[0])
        return fake, real


evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")
ssim_metric.attach(evaluator, "ssim")

fid_values = []
is_values = []
ssim_values = []

previous_model = None


@trainer.on(Events.EPOCH_COMPLETED)
def save_model(engine):
    try:
        os.remove(previous_model)
    except:
        print("No previous saved model found")

    if fid_values and (fid_values[-1] == min(fid_values)):
        MODEL_PATH = f"./models/{model}_G_losses_{G_losses[-1]}_D_losses_{D_losses[-1]}_fid_{fid_values[-1]}_is_{is_values[-1]}.pth"
        torch.save(netG, MODEL_PATH)
        previous_model = MODEL_PATH


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(test_dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics["fid"]
    is_score = metrics["is"]
    ssim_score = metrics["ssim"]
    fid_values.append(fid_score)
    is_values.append(is_score)
    ssim_values.append(ssim_score)
    print(f"Epoch [{engine.state.epoch}/{n_epoch}] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")
    print(f"*  SSIM : {ssim_score:4f}")


# -------------------- Training Visual Setup --------------------

RunningAverage(output_transform=lambda x: x["Loss_G"]).attach(trainer, "Loss_G")
RunningAverage(output_transform=lambda x: x["Loss_D"]).attach(trainer, "Loss_D")

ProgressBar().attach(trainer, metric_names=["Loss_G", "Loss_D"])
ProgressBar().attach(evaluator)

# -------------------- Begin Training --------------------


def training(*args):
    trainer.run(train_dataloader, max_epochs=n_epoch)


with idist.Parallel(backend="nccl") as parallel:
    parallel.run(training)

# -------------------- Model Inference --------------------

# %matplotlib inline

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()

# -------------------- Model Inference --------------------

fig, ax1 = plt.subplots()

plt.title("Evaluation Metric During Training")

color = "tab:red"
ax1.set_xlabel("epochs")
ax1.set_ylabel("IS", color=color)
ax1.plot(is_values, color=color)

ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel("FID", color=color)
ax2.plot(fid_values, color=color)

fig.tight_layout()

# -------------------- Model Inference --------------------

# %matplotlib inline

# Grab a batch of real images from the dataloader
real_batch = next(iter(train_dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),
        (1, 2, 0),
    )
)

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(img_list[-1], padding=2, normalize=True).cpu(),
        (1, 2, 0),
    )
)

# -------------------- Save Model --------------------

# save model
# MODEL_PATH = f"./models/{model}_G_losses_{G_losses[-1]}_D_losses_{D_losses[-1]}_fid_{fid_values[-1]}_is_{is_values[-1]}.pth"
# torch.save(netG, MODEL_PATH)

# load model
# model = torch.load(MODEL_PATH)
# model.eval()
