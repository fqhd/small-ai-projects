import torch
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.models.vgg import vgg19, VGG19_Weights
import numpy as np

device = (
	'cuda' if torch.cuda.is_available() else
	'mps' if torch.backends.mps.is_available() else
	'cpu'
)

to_tensor = ToTensor()
to_pil_image = ToPILImage()

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class CustomInceptionV3(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

		# Extract layers up to Mixed_6a
		self.layer1 = torch.nn.Sequential(
			*net.features[:10]
		)

		self.layer2 = torch.nn.Sequential(
			*net.features[10:12]
		)

	def forward(self, x):
		# Forward pass through the selected layers
		features_a = self.layer1(x)
		features_b = self.layer2(features_a)
		return features_a, features_b

model = CustomInceptionV3()
model = model.to(device)
model.requires_grad_(False)

LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(device)

def dreamify(image, iters, lr):
	# Normalize Image
	image = (image / 255.0).astype('float32')
	image = (image - IMAGENET_MEAN_1) / IMAGENET_STD_1

	image = to_tensor(image)
	image = image.to(device)
	image = image.view(1, *image.shape)
	image.requires_grad = True

	for _ in range(iters):
		# Predict using model
		features_a, features_b = model(image)

		# Calculate the loss(this is what we are trying to maximize)
		loss_1 = torch.mean(features_a)
		loss_2 = torch.mean(features_b)
		loss = torch.mean(torch.stack([loss_1, loss_2]))
		
		# Backpropagate this loss(aka compute the gradients of the loss with respect to the image)
		loss.backward()

		# Pring loss(Optional)
		print(f'Loss: {loss.item():.3f}')

		grad = image.grad.data

		# Normalize the gradients
		g_std = torch.std(grad)
		g_mean = torch.mean(grad)
		grad = grad - g_mean
		grad = grad / g_std

		# Add gradients to image
		image.data += grad * lr

		# Clear gradients and clamp image
		image.grad.data.zero_()
		image.data = torch.max(torch.min(image, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)
	image = np.moveaxis(image.to('cpu').detach().numpy()[0], 0, 2)

	# Denormalize Image
	image = (image * IMAGENET_STD_1) + IMAGENET_MEAN_1
	image = np.clip(image, 0.0, 1.0)
	image = (image * 255.0).astype('uint8')
	
	return image