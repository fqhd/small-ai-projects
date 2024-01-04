import torch
from torch import nn
from torchvision.models.vgg import vgg16

class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg = vgg16(weights='IMAGENET1K_V1').features

		for param in self.vgg.parameters():
			param.requires_grad = False
		
		self.fc = nn.Sequential(
			nn.Flatten(),

			nn.Linear(512 * 7 * 7, 4096),
			nn.Dropout(0.5),
			nn.ReLU(True),
			
			nn.Linear(4096, 4096),
			nn.Dropout(0.5),
			nn.ReLU(True),

			nn.Linear(4096, 4096),
			nn.Dropout(0.5),
			nn.ReLU(True),

			nn.Linear(4096, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		features = self.vgg(x)
		out = self.fc(features)
		return out
	
if __name__ == '__main__':
	img = torch.randn(32, 3, 224, 224)

	print(img.shape)

	net = Network()

	out = net(img)

	print(out.shape)