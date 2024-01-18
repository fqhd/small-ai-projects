from training import train
from utils import getDataloader
import torch
from network import EncoderRNN, AttnDecoderRNN
from globals import hidden_size

batch_size = 32

device = (
	'cuda' if torch.cuda.is_available() else
	'mps' if torch.backends.mps.is_available() else
	'cpu'
)

input_lang, output_lang, train_dataloader = getDataloader(batch_size, device)

encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)

encoder = encoder.to(device)
decoder = decoder.to(device)

train(train_dataloader, encoder, decoder, 80, device)

encoder.cpu()
decoder.cpu()

torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')