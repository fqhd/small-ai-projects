from tqdm import tqdm
from torch import optim
from torch import nn

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
	total_loss = 0
	for data in tqdm(dataloader):
		input_tensor, target_tensor = data
		input_tensor = input_tensor.to(device)
		target_tensor = target_tensor.to(device)

		encoder_outputs, encoder_hidden = encoder(input_tensor)
		decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, device, target_tensor=target_tensor)

		loss = criterion(
			decoder_outputs.view(-1, decoder_outputs.size(-1)),
			target_tensor.view(-1)
		)

		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()
		loss.backward()
		encoder_optimizer.step()
		decoder_optimizer.step()

		total_loss += loss.item()

	return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, device, learning_rate=1e-3):
	losses = []

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
	criterion = nn.NLLLoss()

	for epoch in range(1, n_epochs+1):
		loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
		print(f'Epoch {epoch}) Loss: {loss}')
		losses.append(loss)

	return losses	
