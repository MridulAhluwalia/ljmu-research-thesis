import logging
import os
import random
from math import floor, log2

import PIL.Image as Image

import ignite
import ignite.distributed as idist
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import FID, SSIM, InceptionScore, RunningAverage
from model import Discriminator, Generator
from torchvision.datasets import ImageFolder
from train import get_loader
from utils import gradient_penalty, seed_everything

# -------------------- Initialization --------------------

seed_everything(seed=999)

ignite.utils.setup_logger(
    name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING
)
ignite.utils.setup_logger(
    name="ignite.distributed.launcher.Parallel", level=logging.WARNING
)

# -------------------- Configuration --------------------

START_TRAIN_AT_IMG_SIZE = 4
Z_DIM = 256
N_CHANNELS = 3
FEATURE_DIM = 256

experiment_num = 1
save_threshold = 1

c_lambda = 10
crit_repeats = 1

lr = 1e-3
beta_1 = 0
beta_2 = 0.99

BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)

fundus_type = "N"
model_name = f"PGGAN_{fundus_type}"
input_data_path = f"../processed/{fundus_type}"
analysis_path = f"../analysis/{model_name}/{experiment_num}"

fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=idist.device())

# -------------------- Load Dataset --------------------


def load_data(image_size):
    global load_batch_size, train_dataset
    load_batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    data_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    total_imgs = len(os.listdir(f"{input_data_path}/left/")) + len(
        os.listdir(f"{input_data_path}/right/")
    )
    test_batch_len = 2 ** floor(log2(total_imgs))

    train_dataset = ImageFolder(root=input_data_path, transform=data_transform)
    test_dataset = torch.utils.data.Subset(
        train_dataset, torch.arange(test_batch_len)
    )

    train_dataloader = idist.auto_dataloader(
        train_dataset,
        batch_size=load_batch_size,
        num_workers=2,
        shuffle=True,
    )

    test_dataloader = idist.auto_dataloader(
        test_dataset,
        batch_size=load_batch_size,
        num_workers=2,
        shuffle=False,
    )
    return train_dataloader, test_dataloader


# -------------------- Load Models --------------------


gen = Generator(Z_DIM, FEATURE_DIM, img_channels=N_CHANNELS).to(idist.device())

critic = Discriminator(Z_DIM, FEATURE_DIM, img_channels=N_CHANNELS).to(
    idist.device()
)

# initialize optimizers and scalers for FP16 training
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta_1, beta_2))


# -------------------- Training Setup --------------------


scaler_critic = torch.cuda.amp.GradScaler()
scaler_gen = torch.cuda.amp.GradScaler()


# -------------------- Training Setup --------------------


def training_step(engine, batch):
    global alpha, step, train_dataset, batch_size
    gen.train()
    critic.train()

    real, _ = batch
    batch_size = real.shape[0]

    real = real.to(idist.device())
    noise = torch.randn(batch_size, Z_DIM, 1, 1, device=idist.device())

    with torch.cuda.amp.autocast():
        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(
            critic, real, fake, alpha, step, device=idist.device()
        )
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + c_lambda * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )

    opt_critic.zero_grad()
    scaler_critic.scale(loss_critic).backward()
    scaler_critic.step(opt_critic)
    scaler_critic.update()

    D_losses.append(loss_critic)

    # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
    with torch.cuda.amp.autocast():
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

    opt_gen.zero_grad()
    scaler_gen.scale(loss_gen).backward()
    scaler_gen.step(opt_gen)
    scaler_gen.update()

    G_losses.append(loss_gen.item())

    # Update alpha and ensure less than 1
    alpha += batch_size / (
        (PROGRESSIVE_EPOCHS[step] * 0.5) * len(train_dataset)
    )
    alpha = min(alpha, 1)

    return {
        "Loss_G": loss_gen.item(),
        "Loss_D": loss_critic.item(),
        "D_x": critic_real.mean().item(),
        "D_G_z1": critic_fake.mean().item(),
        "D_G_z2": gen_fake.mean().item(),
    }


trainer = Engine(training_step)


# -------------------- Training Setup --------------------


def initialize_fn(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# @trainer.on(Events.STARTED)
def init_weights():
    gen.apply(initialize_fn)
    critic.apply(initialize_fn)


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
    global aplha, step
    with torch.no_grad():
        fake = gen(fixed_noise, alpha, step).cpu()
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
ssim_metric = SSIM(data_range=1.0, device=idist.device())


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def evaluation_step(engine, batch):
    global gReal, gFake, alpha, step, batch_size_2
    #     print("test batch_size: ", batch_size)
    gReal, _ = batch
    gReal = gReal.to(idist.device())
    with torch.no_grad():
        noise = torch.randn(batch_size_2, Z_DIM, 1, 1, device=idist.device())
        gFake = gen(noise, alpha, step)
        fake = interpolate(gFake)
        real = interpolate(gReal)
        return fake, real


evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")
# ssim_metric.attach(evaluator, "ssim")

previous_model = None

fid_values = []
is_values = []
ssim_values = []

# @trainer.on(Events.EPOCH_COMPLETED)
def save_model(engine):
    global previous_model, MODEL_PATH
    if engine.state.epoch < save_threshold:
        return
    if fid_values and (
        fid_values[-1] == min(fid_values[-(save_threshold - 1) :])
    ):
        print("Saving new model")
        MODEL_PATH = f"./models/{model_name}_n_epoch_{engine.state.epoch}_G_losses_{G_losses[-1]:4f}_D_losses_{D_losses[-1]:4f}_fid_{fid_values[-1]:4f}_is_{is_values[-1]:4f}_ssim_{ssim_values[-1]:4f}.pth"
        torch.save(netG, MODEL_PATH)
        if previous_model:
            print("Removing previous model")
            os.remove(previous_model)
        previous_model = MODEL_PATH


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    global gReal, gFake, test_dataloader, image_size, step

    if image_size <= 8:
        f"Epoch [{engine.state.epoch}/{PROGRESSIVE_EPOCHS[step]}] Metric Scores for Image Size: {image_size}"
        print(f"*  G_losses : {G_losses[-1]:4f}")
        print(f"*  D_losses : {D_losses[-1]:4f}")
        return

    evaluator.run(test_dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics["fid"]
    is_score = metrics["is"]
    fid_values.append(fid_score)
    is_values.append(is_score)

    ssim_metric = SSIM(data_range=1.0, device=idist.device())
    ssim_metric.update((gFake, gReal))
    ssim_score = float(ssim_metric.compute())
    ssim_values.append(ssim_score)

    print(
        f"Epoch [{engine.state.epoch}/{PROGRESSIVE_EPOCHS[step]}] Metric Scores for Image Size: {image_size}"
    )
    print(f"*       FID : {fid_score:4f}")
    print(f"*        IS : {is_score:4f}")
    print(f"*      SSIM : {ssim_score:4f}")
    print(f"*  G_losses : {G_losses[-1]:4f}")
    print(f"*  D_losses : {D_losses[-1]:4f}")


# -------------------- Progress Bar --------------------

RunningAverage(output_transform=lambda x: x["Loss_G"]).attach(trainer, "Loss_G")
RunningAverage(output_transform=lambda x: x["Loss_D"]).attach(trainer, "Loss_D")

ProgressBar().attach(trainer, metric_names=["Loss_G", "Loss_D"])
ProgressBar().attach(evaluator)


# -------------------- Training Step --------------------


def training(*args):
    global alpha, step, test_dataloader, image_size
    start_step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    for step in range(start_step, 7):
        alpha = 1e-5
        image_size = 4 * 2 ** step
        tarin_loader, test_dataloader = load_data(image_size)
        trainer.run(train_loader, max_epochs=PROGRESSIVE_EPOCHS[step])


with idist.Parallel(backend="nccl") as parallel:
    parallel.run(training)
