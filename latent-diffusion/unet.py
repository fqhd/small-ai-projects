import torch
from torch import nn
from modules import DoubleConv, Down, SelfAttention, Up

class UNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.time_dim = 256
		n_features = 24
		att_features = 32
		self.inc = DoubleConv(3, n_features)
		self.down1 = Down(n_features, n_features*2)
		self.sa1 = SelfAttention(n_features*2, att_features)
		self.down2 = Down(n_features*2, n_features*4)
		self.sa2 = SelfAttention(n_features*4, att_features//2)
		self.down3 = Down(n_features*4, n_features*4)
		self.sa3 = SelfAttention(n_features*4, 8)

		self.bot1 = DoubleConv(n_features*4, n_features*8)
		self.bot2 = DoubleConv(n_features*8, n_features*8)
		self.bot3 = DoubleConv(n_features*8, n_features*4)

		self.up1 = Up(n_features*8, n_features*2)
		self.sa4 = SelfAttention(n_features*2, att_features//2)
		self.up2 = Up(n_features*4, n_features)
		self.sa5 = SelfAttention(n_features, att_features)
		self.up3 = Up(n_features*2, n_features)
		self.sa6 = SelfAttention(n_features, att_features * 2)
		self.outc = nn.Conv2d(n_features, 3, kernel_size=1)

	def sinusodial_embedding(self, t, channels, device):
		inv_freq = 1.0 / (
			10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
		)
		pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
		pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
		pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
		return pos_enc
	
	def forward(self, x, t, device):
		t = t.unsqueeze(-1).type(torch.float)
		t = self.sinusodial_embedding(t, self.time_dim, device)

		x1 = self.inc(x)
		x2 = self.down1(x1, t)
		x2 = self.sa1(x2)
		x3 = self.down2(x2, t)
		x3 = self.sa2(x3)
		x4 = self.down3(x3, t)
		x4 = self.sa3(x4)

		x4 = self.bot1(x4)
		x4 = self.bot2(x4)
		x4 = self.bot3(x4)


		x = self.up1(x4, x3, t)
		x = self.sa4(x)
		x = self.up2(x, x2, t)
		x = self.sa5(x)
		x = self.up3(x, x1, t)
		x = self.sa6(x)
		output = self.outc(x)
		return output
