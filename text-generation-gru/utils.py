import torch

def one_hot_sequence_batch(vocab, sequences):
	t = torch.zeros(len(sequences), len(sequences[0]), len(vocab))
	for i, seq in enumerate(sequences):
		for j, c in enumerate(seq):
			t[i][j][vocab.index(c)] = 1
	return t

def one_hot_sequence(vocab, sequence):
	t = torch.zeros(len(sequence), len(vocab))
	for i, c in enumerate(sequence):
		t[i][vocab.index(c)] = 1
	return t

def index_sequence_batch(vocab, sequences):
	t = torch.empty(size=(len(sequences), len(sequences[0])), dtype=torch.int32)
	for i, seq in enumerate(sequences):
		for j, c in enumerate(seq):
			t[i][j] = int(vocab.index(c))
	return t