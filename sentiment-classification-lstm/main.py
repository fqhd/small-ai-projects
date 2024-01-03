import torch
from torch import nn
from model import Net
from data import random_training_example, random_testing_example
from helper import standardize_text, tokenize_text, vectorize_word_tokens
import matplotlib.pyplot as plt

device = (
	'cuda' if torch.cuda.is_available() else
	'mps' if torch.backends.mps.is_available() else
	'cpu'
)

embedding_dim = 16
output_dim = 1
hidden_dim = 128
num_layers = 2
learning_rate = 2e-4
iterations_per_epoch = 100
num_epochs = 100
batch_size = 32
sequence_length = 50

model = Net(embedding_dim, num_layers, hidden_dim, output_dim)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

def get_batch(batch_size, seq_length, subset='training'):
	b_inputs = []
	b_labels = []
	for _ in range(batch_size):
		if subset == 'training':
			example, label = random_training_example()
		else:
			example, label = random_testing_example()
		example = standardize_text(example)
		example = tokenize_text(example)
		example = vectorize_word_tokens(example, seq_length)
		b_inputs.append(example)
		b_labels.append([label])
	return torch.tensor(b_inputs).int(), torch.tensor(b_labels).float()

def test():
	average_loss = 0
	average_accuracy = 0
	for _ in range(iterations_per_epoch):
		example, label = get_batch(batch_size, sequence_length, subset='testing')
		example = example.to(device)
		label = label.to(device)
		
		with torch.no_grad():
			prediction = model(example, device)

		loss = loss_fn(prediction, label)
		average_loss += loss.item()
		average_accuracy += (prediction.round() == label).float().sum().item() / len(label)
	average_loss /= iterations_per_epoch
	average_accuracy /= iterations_per_epoch
	return average_loss, average_accuracy

train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []
for epoch in range(num_epochs):
	average_loss = 0
	average_accuracy = 0
	for i in range(iterations_per_epoch):
		example, label = get_batch(batch_size, sequence_length)
		example = example.to(device)
		label = label.to(device)

		prediction = model(example, device)

		loss = loss_fn(prediction, label)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		average_loss += loss.item()
		average_accuracy += (prediction.round() == label).float().sum().item() / len(label)
	average_loss /= iterations_per_epoch
	average_accuracy /= iterations_per_epoch
	train_loss.append(average_loss)
	train_accuracy.append(average_accuracy)

	test_l, test_acc = test()
	test_loss.append(test_l)
	test_accuracy.append(test_acc)

	current_idx = (epoch+1)*iterations_per_epoch
	total_iters = iterations_per_epoch*num_epochs
	progress = current_idx / total_iters
	progress *= 100
	progress = int(progress)
	print(f'EPOCH {epoch+1}) {progress}% Train Loss: {average_loss:.2f} Train Accuracy: {average_accuracy*100:.2f} Test Accuracy: {test_acc*100:.2f}')


plt.figure()
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(test_accuracy, label='Testing Accuracy')
plt.legend()
plt.show()


def predict(sentence):
	sentence = standardize_text(sentence)
	sentence = tokenize_text(sentence)
	sentence = vectorize_word_tokens(sentence, len(sentence))
	inputs = torch.tensor([sentence]).to(device)
	with torch.no_grad():
		prediction = model(inputs, device)
	return prediction[0].item()

while True:
	sentence = input('Input: ')
	if sentence == 'q':
		break
	prediction = predict(sentence)
	if prediction > 0.5:
		print('Positive')
	else:
		print('Negative')
