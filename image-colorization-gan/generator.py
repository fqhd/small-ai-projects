import torch
from torch import nn
from network import *
from constants import *

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.project = nn.Linear(Z_DIM, IMG_SIZE*IMG_SIZE)
		self.unet = AttU_Net(2, 3, n_features=64)
	
	def forward(self, x, z):
		b_size = z.size(0)
		z = self.project(z)
		z = torch.reshape(z, (b_size, 1, IMG_SIZE, IMG_SIZE))
		x = torch.concat((x, z), dim=1)
		return self.unet(x)
	
if __name__ == '__main__':
	x = torch.randn(32, 1, 64, 64)

	model = Generator()

	trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad
	)
	
	print('Trainable Params:', trainable_params)

	out = model(x)

	print(out.shape)