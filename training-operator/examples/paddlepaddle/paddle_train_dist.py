# Distributed training example for Paddle Paddle
import os
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.vision.datasets import MNIST

# Set CPU_NUM environment variable to specify the number of CPUs
os.environ["CPU_NUM"] = "4"

# Initialize parallel environment
paddle.distributed.init_parallel_env()

# Define a simple feed-forward neural network
class SimpleNN(nn.Layer):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.reshape([-1, 28*28])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and use DataParallel for distributed training
model = SimpleNN()
model = paddle.DataParallel(model)

# Create DataLoader for MNIST dataset
train_dataset = MNIST(mode='train')
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(parameters=model.parameters(), learning_rate=0.001)

# Training loop
for epoch in range(5):
    for batch_id, data in enumerate(train_loader):
        images, labels = data
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if batch_id % 100 == 0:
            print(f"Epoch [{epoch + 1}/5], Step [{batch_id + 1}/{len(train_loader)}], Loss: {loss.item()}")

print("Training Complete!"