import torch
import torch.utils.data as dutils
from dataset import PlanetDataset
from unet import UNet
from tqdm import tqdm
from diffusion_tools import *

dataset = PlanetDataset()
dataloader = dutils.DataLoader(dataset, batch_size=4, shuffle=True)

device = 'cuda'

model = UNet().to(device)
num_params = 0
for param in model.parameters():
	num_params += param.numel()
print(f'Total Params: {num_params}')

mse = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def train():
	average_loss = 0
	for images in tqdm(dataloader):
		images = images.to(device)
		t = sample_timesteps(images.shape[0], device)
		x_t, noise = noise_images(images, t)
		predicted_noise = model(x_t, t, device)
		loss = mse(noise, predicted_noise)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		average_loss += loss.item()
	average_loss /= len(dataloader)
	return average_loss

losses = []
for epoch in range(20):
	loss = train()
	losses.append(loss)
	print(f'Epoch {epoch}) Loss: {loss}')

torch.save(model.state_dict(), 'model.pth')

print(losses)