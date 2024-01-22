from dataset import dataloader, test_dl, dataset
from generator import Generator
from discriminator import *
from constants import *
from tqdm import tqdm
from torch import nn
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import time
import matplotlib.pyplot as plt
from network import *

print(f'Started training using device: {device}')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

d_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
g_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
p_g_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)

loss_fn = nn.BCELoss()
recon_loss = nn.L1Loss()

fixed_noise = torch.randn(test_dl.batch_size, Z_DIM, device=device)
fixed_gray_images, _ = next(iter(test_dl))

plt.figure(figsize=(8, 8))
for i in range(16):
	plt.subplot(4, 4, i+1)
	plt.imshow(T.ToPILImage()(fixed_gray_images[i]), cmap='gray')
	plt.axis('off')
plt.show()

fixed_small_images = fixed_gray_images.to(device)

g_losses = []
d_losses = []

start = time.time()
idx = 0
p_value = 0

g_pretrain_loss = 1
while g_pretrain_loss > 0.04:
	avg_loss = 0
	for _ in range(10):
		g_batch, _ = next(iter(dataloader))
		g_batch = g_batch.to(device)

		noise = torch.randn(BATCH_SIZE, Z_DIM, device=device)
		predict = generator(g_batch, noise)
		g_curr_loss = recon_loss(predict, g_batch.expand(-1, 3, -1, -1))
		generator.zero_grad()
		g_curr_loss.backward()
		p_g_opt.step()
		
		avg_loss += g_curr_loss.item()
	avg_loss /= 10
	g_pretrain_loss = avg_loss

print('Got satisfactory loss for generator')


p = 1
for epoch in range(EPOCHS):
	avg_d_loss = 0
	avg_g_loss = 0
	for gray_batch, color_batch in dataloader:
		gray_batch = gray_batch.to(device)
		color_batch = color_batch.to(device)
		c3_gray_batch = gray_batch.expand(-1, 3, -1, -1)

		b_size = gray_batch.size(0)

		color_batch = color_batch + (c3_gray_batch - color_batch) * p

		# Train on Real
		discriminator.zero_grad()
		y_hat_real = discriminator(color_batch).view(-1)
		y_real = torch.ones_like(y_hat_real, device=device)
		real_loss = loss_fn(y_hat_real, y_real)
		real_loss.backward()

		# Train on Fake
		noise = torch.randn(b_size, Z_DIM, device=device)
		fake_images = generator(gray_batch, noise)
		y_hat_fake = discriminator(fake_images.detach()).view(-1)
		y_fake = torch.zeros_like(y_hat_fake)
		fake_loss = loss_fn(y_hat_fake, y_fake)
		fake_loss.backward()
		d_opt.step()

		p -= 0.00005
		if p < 0:
			p = 0

		# Train generator
		generator.zero_grad()
		y_hat_fake = discriminator(fake_images)
		adversarial_loss = loss_fn(y_hat_fake, torch.ones_like(y_hat_fake))
		gray_fake = dataset.gray(fake_images)
		reconstruction_loss = recon_loss(gray_fake, gray_batch)
		g_loss = 1e-3 * adversarial_loss + reconstruction_loss
		g_loss.backward()
		g_opt.step()

		# Log losses
		avg_g_loss += adversarial_loss.item()
		avg_d_loss += (real_loss.item() + fake_loss.item()) / 2
		if idx % 10 == 0:
			avg_d_loss /= 10
			avg_g_loss /= 10
			d_losses.append(avg_d_loss)
			g_losses.append(avg_g_loss)
			print(f'G_Loss: {avg_g_loss} D_Loss: {avg_d_loss}')
			avg_g_loss = 0
			avg_d_loss = 0
			print(p)
		idx += 1
	
	with torch.no_grad():
		predicted_images = generator(fixed_small_images, fixed_noise)[:16]
	img = T.ToPILImage()(vutils.make_grid(predicted_images.to('cpu'), padding=2, nrow=4))
	img.save(f'progress/epoch_{epoch}.jpg')
	generator = generator.to('cpu')
	torch.save(generator, f'models/generator_epoch_{epoch}.pkl')
	generator = generator.to(device)

plt.figure(figsize=(8, 5))
plt.plot(g_losses, label='G_Loss')
plt.plot(d_losses, label='D_Loss')
plt.title('Generator and Discriminator Loss')
plt.legend()
plt.show()

train_time = time.time() - start
print(f'Total training time: {train_time // 60} minutes')