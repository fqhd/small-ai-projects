import re
import unicodedata
import random
from globals import *

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
	return s.strip()


def readLangs():
	print("Reading lines...")

	# Read the file and split into lines
	lines = open('data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')

	# Split every line into pairs and normalize
	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

	input_lang = Lang('English')
	output_lang = Lang('French')

	return input_lang, output_lang, pairs


def filterPair(p):
	eng_prefixes = (
		"i am ", "i m ",
		"he is", "he s ",
		"she is", "she s ",
		"you are", "you re ",
		"we are", "we re ",
		"they are", "they re "
	)

	return len(p[0].split(' ')) < MAX_LENGTH and \
		len(p[1].split(' ')) < MAX_LENGTH and \
		p[0].startswith(eng_prefixes)


def prepareData():
	input_lang, output_lang, pairs = readLangs()
	print("Read %s sentence pairs" % len(pairs))
	pairs = [p for p in pairs if filterPair(p)]
	print("Trimmed to %s sentence pairs" % len(pairs))
	print("Counting words...")
	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	print("Counted words:")
	print(input_lang.name, input_lang.n_words)
	print(output_lang.name, output_lang.n_words)
	return input_lang, output_lang, pairs

if __name__ == '__main__':
	input_lang, output_lang, pairs = prepareData()
	print(random.choice(pairs))