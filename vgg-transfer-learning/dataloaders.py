import os
import torch.utils.data as dutils
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
import torchvision.transforms as T
import random

TRAIN_SPLIT = 0.8

class Dataset(dutils.Dataset):
	def __init__(self, root, transform=None, split='training'):
		super().__init__()
		self.root = root
		self.transform = transform

		good_files = os.listdir(f'{root}/good')
		bad_files = os.listdir(f'{root}/bad')
		arr = []
		for gf in good_files:
			arr.append((gf, 1))
		for bf in bad_files:
			arr.append((bf, 0))
		random.shuffle(arr)
		num_files = len(arr)

		num_train = int(num_files * TRAIN_SPLIT)

		if split == 'training':
			self.arr = arr[:num_train]
		elif split == 'testing':
			self.arr = arr[num_train:]
		else:
			print('Unknown split, using full dataset')
			self.arr = arr

		print(f'Using {len(self.arr)} files for {split}')

		self.resize = T.Resize(224, antialias=True)

	def __len__(self):
		return len(self.arr)
	
	def __getitem__(self, idx):
		path, label = self.arr[idx]
		if label:
			img = read_image(f'{self.root}/good/{path}')
		else:
			img = read_image(f'{self.root}/bad/{path}')
		if self.transform:
			img = self.transform(img)
		else:
			img = self.resize(img)
		return img / 255.0, label

transform = T.Compose([
	T.Resize((320, 320), antialias=True),
	T.RandomRotation(20),
	T.CenterCrop((256, 256)),
	T.RandomCrop((224, 224))
])

train_ds = Dataset('data', transform=transform, split='training')
test_ds = Dataset('data', split='testing')

train_dl = dutils.DataLoader(train_ds, batch_size=4, shuffle=True)
test_dl = dutils.DataLoader(test_ds, batch_size=32, shuffle=True)

if __name__ == '__main__':
	images, labels = next(iter(test_dl))
	class_names = ['bad', 'good']

	plt.figure(figsize=(8, 8))
	for i in range(16):
		plt.subplot(4, 4, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(T.ToPILImage()(images[i]))
		plt.xlabel(class_names[labels[i]])
	plt.show()
