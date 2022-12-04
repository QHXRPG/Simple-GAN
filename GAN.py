import torch
import torchvision
import torch.nn as nn
import numpy as np

image_size = [1, 28, 28]
latent_dim = 96
batch_size = 64
use_gpu = torch.cuda.is_available()
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                             #  torchvision.transforms.Normalize([0.5], [0.5]),
                                         ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, 768),
            nn.Sigmoid(),
        )

    def forward(self, z):
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)
        return image


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        prob = self.model(image.reshape(image.shape[0], -1))
        return prob

generator = Generator()
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)


num_epoch = 200
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)

        g_optimizer.zero_grad()
        recons_loss = torch.abs(fake_images-gt_images).mean()
        g_loss = recons_loss*0.05 + loss_fn(discriminator(fake_images), labels_one)
        g_loss.backward()
        g_optimizer.step()


        d_optimizer.zero_grad()
        real_loss = loss_fn(discriminator(gt_images), labels_one)
        fake_loss = loss_fn(discriminator(fake_images.detach()), labels_zero)
        d_loss = (real_loss + fake_loss)
        # 观察real_loss与fake_loss，同时下降同时达到最小值，并且差不多大，说明D已经稳定了
        d_loss.backward()
        d_optimizer.step()

        if i % 50 == 0:
            print(f"step:{len(dataloader)*epoch+i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if i % 400 == 0:
            image = fake_images[:16].data
            torchvision.utils.save_image(image, f"image_{len(dataloader)*epoch+i}.png", nrow=4)