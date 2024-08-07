import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Datasets
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data Loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

model = Net()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with timing
start_time = time.time()
for epoch in range(2):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch+1}/{2}")
            print(f"Batch {batch_idx}/{len(trainloader)} [{'=' * (batch_idx // 100) + '>' + '.' * (18 - (batch_idx // 100))}] - loss: {loss.item():.4f} - accuracy: {output.max(1)[1].eq(labels).sum().item() / labels.size(0):.4f}")

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

total_training_time = time.time() - start_time
print(f"Total Training Time: {total_training_time:.2f} seconds")
print("Model finished training.")

# Evaluation
model.eval()
correct = 0
total = 0
test_loss = 0.0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(testloader)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {100 * correct / total:.2f}%")
