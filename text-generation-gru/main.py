import torch
from torch import nn
from data import get_vocab, get_next_sequence, split_sequence
from utils import one_hot_sequence, index_sequence_batch
from tqdm import tqdm
from text_gru import TextGRU
from constants import *

vocab = get_vocab()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device: {device}')

model = TextGRU()
model = model.to(device)

num_params = 0
for param in model.parameters():
	num_params += param.numel()
print(f'Total Params: {num_params}')

criterion = nn.CrossEntropyLoss()
adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
	print('Epoch:', epoch)
	avg_loss = 0
	for _ in tqdm(range(iterations)):
		sequences = [get_next_sequence(sequence_length+1) for _ in range(batch_size)]
		inputs, target = [], ''

		for seq in sequences:
			i, t = split_sequence(seq)
			inputs.append(i)
			target += t

		inputs = index_sequence_batch(vocab, inputs)
		target = one_hot_sequence(vocab, target)

		inputs = inputs.to(device)
		target = target.to(device)

		preds = model(inputs, device)

		loss = criterion(preds, target)

		model.zero_grad()
		loss.backward()
		adam.step()

		avg_loss += loss.item()
	avg_loss /= iterations
	print('Loss:', avg_loss)

model = model.cpu()
torch.save(model.state_dict(), 'model.pth')