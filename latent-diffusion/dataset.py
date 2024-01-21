import torchvision.transforms as T
import torch.utils.data as dutils
import os
from torchvision.io.image import read_image
import torch

class PlanetDataset(dutils.Dataset):
	def __init__(self):
		super().__init__()
		self.paths = os.listdir('data/images')
		self.transform = T.Compose([
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, idx):
		img = read_image('data/images/'+self.paths[idx])
		img = img / 255.0
		return self.transform(img)
	

if __name__ == '__main__':
	ds = PlanetDataset()
	dl = dutils.DataLoader(ds, batch_size=32, shuffle=True)
	imgs = next(iter(dl))
	print(torch.min(imgs))
	print(torch.max(imgs))
	print(imgs.dtype)
	print(imgs.shape)