import torch
from dataloaders import train_dl, test_dl
from network import Network
import time
import matplotlib.pyplot as plt

device = (
	'cuda' if torch.cuda.is_available() else
	'mps' if torch.backends.mps.is_available() else
	'cpu'
)

net = Network()
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

def train():
	average_loss = 0
	average_accuracy = 0
	for images, labels in train_dl:
		images = images.to(device)
		labels = labels.float().to(device)

		pred = net(images)
		pred = pred.view(-1)

		loss = criterion(pred, labels)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		average_loss += loss.item()

		accuracy = (pred.round() == labels).float().sum().item() / len(labels)
		average_accuracy += accuracy
	average_loss /= len(train_dl)
	average_accuracy /= len(train_dl)
	return average_loss, average_accuracy

def test():
	average_loss = 0
	average_accuracy = 0
	for images, labels in test_dl:
		images = images.to(device)
		labels = labels.float().to(device)

		with torch.no_grad():
			pred = net(images)
		pred = pred.view(-1)

		loss = criterion(pred, labels)
		
		average_loss += loss.item()

		accuracy = (pred.round() == labels).float().sum().item() / len(labels)
		average_accuracy += accuracy
	average_loss /= len(test_dl)
	average_accuracy /= len(test_dl)
	return average_loss, average_accuracy

highest_accuracy = 0
start = time.time()
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(50):

	# Train and test the model
	train_loss, train_accuracy = train()
	test_loss, test_accuracy = test()

	# Log training info
	print(f'EPOCH {epoch+1}) {train_loss:.4f} {train_accuracy:.4f} {test_accuracy:.4f}')

	# Keep track of model training steps
	train_losses.append(train_loss)
	train_accuracies.append(train_accuracy)
	test_losses.append(test_loss)
	test_accuracies.append(test_accuracy)

	# Save model every time new highest accuracy is reached
	if test_accuracy > highest_accuracy:
		highest_accuracy = test_accuracy
		print('Found new highest accuracy, saving model...')
		net = net.to('cpu')
		torch.save(net.state_dict(), 'weights.pth')
		net = net.to(device)

train_time = time.time() - start
print(f'Total train time: {train_time} seconds')

plt.figure()

plt.subplot(1, 2, 1)
plt.title('Training and Validation Loss')
plt.plot(train_losses, ls='dashed', label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Training and Validation Accuracy')
plt.plot(train_losses, ls='dashed', label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

