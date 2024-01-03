import string
import os
from unidecode import unidecode
import json

embeddings = {}

def get_word_embedding(word):
	if word in embeddings:
		return embeddings[word]
	return 0

def vectorize_word_tokens(tokens, n_tokens):
	v = [get_word_embedding(word) for word in tokens]
	while len(v) < n_tokens:
		v.append(0)
	return v[:n_tokens]

def standardize_text(text):
	text = text.replace('<br />', ' ')
	text = text.translate(str.maketrans(string.punctuation, ''.join([' ' for _ in string.punctuation]), string.punctuation))
	text = text.lower()
	return text

def tokenize_text(text):
	return text.split(' ')

def top_n_words(dictionary, n):
	# Sort the dictionary items by their values in descending order
	sorted_words = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

	# Take the top n items
	top_n = sorted_words[:n]

	# Extract only the words from the result
	top_n_words = [word for word, count in top_n]

	return top_n_words

def most_common_words(n):
	word_popularity = {}
	for c in ['data/train/neg', 'data/train/pos']:
		for p in os.listdir(c):
			with open(f'{c}/{p}', encoding='utf-8') as f:
				text = f.read()
				text = unidecode(text)
				text = standardize_text(text)
				text = tokenize_text(text)
				for word in text:
					if word in word_popularity:
						word_popularity[word] += 1
					else:
						word_popularity[word] = 1
	return top_n_words(word_popularity, n)

print('Finding most common words in dataset...')
vocab = most_common_words(9999)
print('Creating embedding dictionary...')
embeddings = {word: (i+1) for i, word in enumerate(vocab)}
print('Done.')

if __name__ == '__main__':
	"""
	text = 'Blake Edwards\' legendary fiasco, begins to seem pointless after just 10 minutes. A combination of The Eagle Has Landed, Star!, Oh!'
	print(text)
	print(' ')
	text = standardize_text(text)
	print(text)
	print(' ')
	tokens = tokenize_text(text)
	print(tokens)
	print(' ')
	vectorized = vectorize_word_tokens(tokens)
	print(vectorized)
	"""

	print(most_common_words(100))

