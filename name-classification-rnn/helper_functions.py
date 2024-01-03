import unicodedata
import string
import os
import torch
import random

ALL_LETTERS = string.ascii_letters + ' .,;\''
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
	)

def letter_to_index(letter):
	return ALL_LETTERS.index(letter)

def line_to_tensor(line):
	t = torch.zeros(len(line), 1, N_LETTERS)
	for i, letter in enumerate(line):
		t[i][0][letter_to_index(letter)] = 1
	return t

def letter_to_tensor(l):
	idx = letter_to_index(l)
	t = torch.zeros(1, N_LETTERS)
	t[0][idx] = 1
	return t

def load_data():
	file_paths = os.listdir('data')
	class_names = [x[:-4] for x in file_paths]

	data = {}
	for c_name in class_names:
		with open(f'data/{c_name}.txt', encoding='utf-8') as f:
			data[c_name] = [unicode_to_ascii(x) for x in f.read().split('\n')[:-1]]
	
	return data

def random_training_example(category_lines, all_categories):
	category = random.choice(all_categories)
	line = random.choice(category_lines[category])
	category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
	line_tensor = line_to_tensor(line)
	return category, line, category_tensor, line_tensor


if __name__ == '__main__':
	# print('réasöñ')
	# print(unicode_to_ascii('réasöñ'))

	data = load_data()
	print(data['French'][:5])

	print(letter_to_tensor('j'))
	print(line_to_tensor('Albert'))