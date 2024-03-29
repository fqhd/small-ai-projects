import torch.utils.data as dutils
from constants import *
import torchvision.transforms as T
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt

class Dataset(dutils.Dataset):
	def __init__(self, transform = None):
		self.transform = transform
		self.image_names = os.listdir('dataset')
		self.gray = T.Grayscale()

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		image_path = f'dataset/{self.image_names[idx]}'
		img = read_image(image_path)
		if self.transform:
			img = self.transform(img)
		grayscale = self.gray(img)
		return grayscale, img

transform = T.Compose([
	T.ToPILImage(),
	T.RandomRotation(10),
	T.CenterCrop(164),
	T.RandomCrop(148),
	T.Resize(64),
	T.RandomHorizontalFlip(),
	T.ToTensor()
])

dataset = Dataset(transform=transform)
dataloader = dutils.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dl = dutils.DataLoader(dataset, batch_size=16, shuffle=True)

if __name__ == '__main__':
	img = next(iter(dataloader))

	plt.figure(figsize=(6, 8))
	for i in range(0, 8, 2):
		plt.subplot(4, 2, i+1)
		plt.imshow(T.ToPILImage()(img[0][i]))

		plt.subplot(4, 2, i+2)
		plt.imshow(T.ToPILImage()(img[1][i]))
	plt.show()