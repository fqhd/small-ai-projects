import torch

Z_DIM = 16
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999
IMG_SIZE = 64

device = (
	'cuda' if torch.cuda.is_available() else
	'mps' if torch.backends.mps.is_available() else
	'cpu' 
)