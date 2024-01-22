"""
This is a modified version of code that I took from here:
https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
The modification I made is I added a n_features parameter to the constructor of the unets
This way it is easier to change the size of the unets
I also added a sigmoid activation at the end so the output range is clipped between 0 and 1
"""

import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)

class conv_block(nn.Module):
	def __init__(self,ch_in,ch_out,emb_dim,im_width):
		super(conv_block,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)
		self.project = nn.Linear(emb_dim, im_width*im_width)
		self.im_width = im_width


	def forward(self,x,t):
		b_size = x.size(0)
		projected = self.project(t)
		projected = torch.reshape(projected, (b_size, 1, self.im_width, self.im_width))
		x = torch.cat([x, projected], dim=1)
		x = self.conv(x)
		return x

class up_conv(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(up_conv,self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)

	def forward(self,x):
		x = self.up(x)
		return x


		
class Attention_block(nn.Module):
	def __init__(self,F_g,F_l,F_int):
		super(Attention_block,self).__init__()
		self.W_g = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
			)
		
		self.W_x = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
		)

		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
		
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self,g,x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		psi = self.relu(g1+x1)
		psi = self.psi(psi)

		return x*psi

class AttU_Net(nn.Module):
	def __init__(self,img_ch=3,output_ch=3,n_features=32,time_dim=256,im_width=64):
		super(AttU_Net,self).__init__()

		self.time_dim = time_dim

		self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

		self.Conv1 = conv_block(img_ch+1,n_features,time_dim,im_width)
		self.Conv2 = conv_block(n_features+1,n_features*2,time_dim,im_width//2)
		self.Conv3 = conv_block(n_features*2+1,n_features*4,time_dim,im_width//4)
		self.Conv4 = conv_block(n_features*4+1,n_features*8,time_dim,im_width//8)
		self.Conv5 = conv_block(n_features*8+1,n_features*16,time_dim,im_width//16)

		self.Up5 = up_conv(n_features*16,n_features*8)
		self.Att5 = Attention_block(F_g=n_features*8,F_l=n_features*8,F_int=n_features*4)
		self.Up_conv5 = conv_block(n_features*16+1, n_features*8,time_dim,im_width//8)

		self.Up4 = up_conv(n_features*8,n_features*4)
		self.Att4 = Attention_block(F_g=n_features*4,F_l=n_features*4,F_int=n_features*2)
		self.Up_conv4 = conv_block(n_features*8+1, n_features*4,time_dim,im_width//4)
		
		self.Up3 = up_conv(n_features*4,n_features*2)
		self.Att3 = Attention_block(F_g=n_features*2,F_l=n_features*2,F_int=n_features)
		self.Up_conv3 = conv_block(n_features*4+1, n_features*2,time_dim,im_width//2)
		
		self.Up2 = up_conv(n_features*2,n_features)
		self.Att2 = Attention_block(F_g=n_features,F_l=n_features,F_int=n_features//2)
		self.Up_conv2 = conv_block(n_features*2+1, n_features,time_dim,im_width)

		self.Conv_1x1 = nn.Conv2d(n_features,output_ch,kernel_size=1,stride=1,padding=0)

	def sinusodial_embedding(self, t, channels, device):
		inv_freq = 1.0 / (
			10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
		)
		pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
		pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
		pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
		return pos_enc

	def forward(self,x,t,device):
		t = t.unsqueeze(-1).type(torch.float)
		t = self.sinusodial_embedding(t, self.time_dim, device)

		# encoding path
		x1 = self.Conv1(x, t) # 64x64

		x2 = self.Maxpool(x1)
		x2 = self.Conv2(x2, t) # 32x32
		
		x3 = self.Maxpool(x2)
		x3 = self.Conv3(x3, t) # 16x16

		x4 = self.Maxpool(x3)
		x4 = self.Conv4(x4, t) # 8x8

		x5 = self.Maxpool(x4)
		x5 = self.Conv5(x5, t) # 4x4

		# decoding + concat path
		d5 = self.Up5(x5)
		x4 = self.Att5(g=d5,x=x4)
		d5 = torch.cat((x4,d5),dim=1)        
		d5 = self.Up_conv5(d5, t) # 8x8
		
		d4 = self.Up4(d5)
		x3 = self.Att4(g=d4,x=x3)
		d4 = torch.cat((x3,d4),dim=1)
		d4 = self.Up_conv4(d4, t) # 16x16

		d3 = self.Up3(d4)
		x2 = self.Att3(g=d3,x=x2)
		d3 = torch.cat((x2,d3),dim=1)
		d3 = self.Up_conv3(d3, t) # 32x32

		d2 = self.Up2(d3)
		x1 = self.Att2(g=d2,x=x1)
		d2 = torch.cat((x1,d2),dim=1)
		d2 = self.Up_conv2(d2, t) # 64x64

		d1 = self.Conv_1x1(d2)

		return d1