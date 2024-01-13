import random
from unidecode import unidecode

with open('data/bbt.txt', encoding='utf-8') as f:
	text = unidecode(f.read())

def get_vocab():
	s = set(text)
	vocab = list(s)
	vocab.sort()
	return vocab

sequence_index = 0
def get_next_sequence(seq_length):
	global sequence_index
	sequence = text[sequence_index:sequence_index+seq_length]
	sequence_index += 1
	if sequence_index > len(text)-seq_length:
		sequence_index = 0
	return sequence

def get_random_sequence(seq_length):
	idx = random.randint(0, len(text)-seq_length)
	sequence = text[idx:idx+seq_length]
	return sequence

def split_sequence(sequence):
	return sequence[:-1], sequence[-1]

if __name__ == '__main__':
	print(len(get_vocab()))
	seq_length = 10
	for _ in range(100):
		seq = get_random_sequence(seq_length+1)
		print(seq)