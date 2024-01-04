import torch
from network import Network
from dataloaders import Dataset
import torch.utils.data as dutils
import numpy as np

ds = Dataset('data', transform=None, split='testing')
dl = dutils.DataLoader(ds, batch_size=64, shuffle=False)

device = (
	'cuda' if torch.cuda.is_available() else
	'mps' if torch.backends.mps.is_available() else
	'cpu'
)

net = Network()
net.load_state_dict(torch.load('weights.pth'))
net = net.to(device)
net.eval()

num_iters = 3
num_steps = 10

def custom_round(tensor, threshold):
	tensor.apply_(lambda x: 1.0 if x > threshold else 0.0)

def test(threshold):
	avg_acc = 0
	for imgs, labels in dl:
		imgs = imgs.to(device)
		labels = labels.float().to(device)

		with torch.no_grad():
			pred = net(imgs)
		pred = pred.view(-1)

		pred = pred.to('cpu')
		labels = labels.to('cpu')
		custom_round(pred, threshold)

		acc = (pred == labels).float().sum() / len(labels)
		avg_acc += acc
	avg_acc /= len(dl)
	return avg_acc

def test_threshold(min, max):
	values = np.linspace(min, max, num_steps)
	best_acc = 0
	best_v = 0
	for v in values:
		curr_acc = test(v)
		if curr_acc > best_acc:
			best_v = v
			best_acc = curr_acc
	print(f'Best threshold: {best_v} with accuracy: {best_acc}')
	return best_v

min = 0
max = 1
for i in range(num_iters):
	v = test_threshold(min, max)
	d = max - min
	mid = (max + min) / 2
	min = mid - d / 4
	max = mid + d / 4
