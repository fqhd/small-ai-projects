import torch
from utils import index_sequence_batch
from data import get_vocab
from text_gru import TextGRU
import random
from constants import *

model = TextGRU()
model.load_state_dict(torch.load('model.pth'))
model.eval()

vocab = get_vocab()

def choose_element(probability_dict):
	rand_num = random.uniform(0, 1)
	cumulative_prob = 0

	for element, prob in probability_dict.items():
		cumulative_prob += prob
		if rand_num <= cumulative_prob:
			return element

def predict(text):
	inp = index_sequence_batch(vocab, [text])
	preds = model(inp, 'cpu')[0]
	preds = torch.softmax(preds, dim=0)
	distribution = {}
	for i in range(len(vocab)):
		distribution[vocab[i]] = preds[i].item()
	c = choose_element(distribution)
	return c


while True:
	sequence = input('Input: ')
	num_chars = int(input('Length: '))

	for i in range(num_chars):
		seq_len = min(len(sequence), sequence_length)
		c = predict(sequence[-seq_len:])
		sequence += c

	print('Output:')
	print(sequence)