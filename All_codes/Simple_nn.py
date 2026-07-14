import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer (2 neurons) → Hidden layer (4 neurons)
        self.fc2 = nn.Linear(4, 1)  # Hidden layer (4 neurons) → Output layer (1 neuron)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

model = SimpleNN()
print(model)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Sample Data: (X: inputs, Y: labels)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR function

# Training Loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/1000], Loss: {loss.item():.4f}')

with torch.no_grad():
    test_outputs = model(X)
    print(test_outputs)
