import torch
from tqdm import tqdm

noise_steps = 1000
beta_start = 1e-4
beta_end = 0.02

beta = torch.linspace(beta_start, beta_end, noise_steps, device='cuda')
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)
alpha_hat = alpha_hat.to('cuda')

def noise_images(imgs, t):
	sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
	sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
	eps = torch.randn_like(imgs)
	return sqrt_alpha_hat * imgs + sqrt_one_minus_alpha_hat * eps, eps

def sample_timesteps(n, device):
	return torch.randint(low=1, high=noise_steps, size=(n,), device=device)

def sample_images(model, n, device):
	model.eval()
	with torch.no_grad():
		x = torch.randn((n, 3, 64, 64), device=device)
		for i in tqdm(reversed(range(1, noise_steps))):
			t = (torch.ones(n, device=device) * i).long()
			predicted_noise = model(x, t, device)
			a = alpha[t][:, None, None, None]
			a_hat = alpha_hat[t][:, None, None, None]
			b = beta[t][:, None, None, None]
			if i > 1:
				noise = torch.randn_like(x)
			else:
				noise = torch.zeros_like(x)
			x = 1 / torch.sqrt(a) * (x - ((1 - a) / torch.sqrt(1 - a_hat)) * predicted_noise) + torch.sqrt(b) * noise
	model.train()
	x = torch.clamp(x, -1, 1)
	x = (x + 1) / 2
	x = (x * 255).type(torch.uint8)
	return x