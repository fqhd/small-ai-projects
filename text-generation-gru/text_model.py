import torch
from torch import nn
from data import get_vocab
from constants import *

class TextModel(nn.Module):
	def __init__(self):
		super().__init__()
		vocab_size = len(get_vocab())
		self.rnn_layers = rnn_layers
		self.rnn_units = rnn_units
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.gru = nn.GRU(embedding_dim, rnn_units, rnn_layers, batch_first=True)
		self.dense = nn.Linear(rnn_units, vocab_size)

	def forward(self, x, device):
		h0 = torch.zeros(self.rnn_layers, x.shape[0], self.rnn_units).to(device)

		x = self.embedding(x)
		x, _ = self.gru(x, h0)
		x = x[:, -1, :]
		logits = self.dense(x)

		return logits
	
if __name__ == '__main__':
	network = TextModel()
	x = torch.tensor([[2, 1, 4, 22, 11, 18, 6, 7, 9, 10]])

	logits = network(x, 'cpu')

	print(logits.shape)