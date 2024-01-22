import torch
import torchvision.transforms as T
from dataset import test_dl
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image

generator = torch.load('models/generator_v1.pkl')
generator.eval()

im_names = os.listdir('in')

for image_name in im_names:
	img = read_image(f'in/{image_name}')[:3, :, :]
	img = T.Grayscale()(img) / 255.0
	img = torch.unsqueeze(img, 0)
	out = generator(img)[0]
	out = T.ToPILImage()(out)
	out.save(f'out/{image_name}')