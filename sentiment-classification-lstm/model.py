import torch
from torch import nn

class Net(nn.Module):
	def __init__(self, embedding_dim, num_layers, hidden_dim, output_dim):
		super().__init__()
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim

		self.embedding = nn.Embedding(10000, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
		
		for name, param in self.lstm.named_parameters():
			if 'weight' in name:
				nn.init.xavier_uniform_(param.data)

		# batch_size, seq_length, input_dim
		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(hidden_dim, output_dim),
			nn.Sigmoid()
		)

	def forward(self, x, device):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

		x = self.embedding(x)
		# batch_size, seq_length, embedding_dim

		out, _ = self.lstm(x, (h0, c0))
		# batch_size, seq_length, hidden_dim

		out = out[:, -1, :]
		# batch_size, hidden_dim

		out = self.fc(out)
		# batch_size, output_dim

		return out



if __name__ == '__main__':
	inputs = torch.tensor([[2, 3, 1, 5, 1, 8, 1]])

	model = Net(32, 2, 128, 1)

	with torch.no_grad():
		out = model(inputs, 'cpu')

	print(out.shape)