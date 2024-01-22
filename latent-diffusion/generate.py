import torch
from network import AttU_Net
from diffusion_tools import sample_images
import matplotlib.pyplot as plt
import torchvision.transforms as T

model = AttU_Net()
model = model.to('cuda')
model.load_state_dict(torch.load('model.pth'))

imgs = sample_images(model, 4, 'cuda')

print(imgs.shape)
plt.figure(figsize=(8, 8))
for i in range(4):
	plt.subplot(2, 2, i+1)
	plt.imshow(T.ToPILImage()(imgs[i]))
plt.show()