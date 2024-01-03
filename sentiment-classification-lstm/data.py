import random
import os
from unidecode import unidecode

positive_training_examples = os.listdir('data/train/pos')
negative_training_examples = os.listdir('data/train/neg')
positive_testing_examples = os.listdir('data/test/pos')
negative_testing_examples = os.listdir('data/test/neg')

def random_training_example():
	label = random.randint(0, 1)
	if label == 0:
		example_path = 'data/train/neg/' + random.choice(negative_training_examples)
	else:
		example_path = 'data/train/pos/' + random.choice(positive_training_examples)
	with open(example_path, encoding='utf-8') as f:
		text = f.read()
	return unidecode(text), label


def random_testing_example():
	label = random.randint(0, 1)
	if label == 0:
		example_path = 'data/test/neg/' + random.choice(negative_testing_examples)
	else:
		example_path = 'data/test/pos/' + random.choice(positive_testing_examples)
	with open(example_path, encoding='utf-8') as f:
		text = f.read()
	return unidecode(text), label

if __name__ == '__main__':
	train_example = random_training_example()
	test_example = random_testing_example()
	print(train_example)
	print(' ')
	print(test_example)