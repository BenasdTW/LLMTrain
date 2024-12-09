import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Custom RNN model from scratch (defined earlier)
class SimpleRNNFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNNFromScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight matrices for input to hidden, and hidden to hidden transitions
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size))  # Input to hidden
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))  # Hidden to hidden
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size))  # Hidden to output

        # Bias terms
        self.bh = nn.Parameter(torch.zeros(hidden_size))  # Hidden bias
        self.by = nn.Parameter(torch.zeros(output_size))  # Output bias

    def forward(self, x, hidden):
        outputs = []
        for t in range(x.size(1)):  # Iterate over time steps
            xt = x[:, t, :]  # Input at time step t
            hidden = torch.tanh(xt @ self.Wxh + hidden @ self.Whh + self.bh)  # Update hidden state
            output = hidden @ self.Why + self.by  # Compute output
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # Stack outputs for all time steps
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Step 1: Generate a sine wave dataset
def generate_sine_wave(seq_len, num_samples):
    x = np.linspace(0, 50, num_samples)
    y = np.sin(x)
    data = []
    for i in range(num_samples - seq_len):
        data.append(y[i:i + seq_len + 1])
    data = np.array(data)
    inputs = data[:, :-1]  # Input sequences
    targets = data[:, -1:]  # Target values (next in sequence)
    return inputs, targets

# Generate dataset
seq_len = 10
num_samples = 1000
inputs, targets = generate_sine_wave(seq_len, num_samples)

# Convert to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)  # (samples, seq_len, 1)
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)  # (samples, 1, 1)

# Split into train and test sets
train_size = int(0.8 * len(inputs))
train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
train_targets, test_targets = targets[:train_size], targets[train_size:]

# Step 2: Train the custom RNN
input_size = 1  # Each input is a scalar (1 feature)
hidden_size = 20  # Hidden size for the RNN
output_size = 1  # Output is a scalar (next sine value)
batch_size = 32
num_epochs = 400

# Instantiate the model
model = SimpleRNNFromScratch(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Split training data into batches
    for i in range(0, train_size, batch_size):
        batch_inputs = train_inputs[i:i+batch_size]
        batch_targets = train_targets[i:i+batch_size]

        # Dynamically set the hidden state size to match the batch size
        hidden = model.init_hidden(batch_inputs.size(0))  # Initialize hidden state for the current batch size

        optimizer.zero_grad()  # Zero gradients
        outputs, hidden = model(batch_inputs, hidden.detach())  # Forward pass
        loss = criterion(outputs[:, -1, :], batch_targets)  # Compute loss on last time step
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss/len(train_inputs)}')


# Step 3: Test the model
model.eval()
with torch.no_grad():
    hidden = model.init_hidden(len(test_inputs))
    test_outputs, _ = model(test_inputs, hidden)
    test_loss = criterion(test_outputs[:, -1, :], test_targets)
    print(f'Test Loss: {test_loss.item()}')

# Step 4: Plot the results
plt.plot(test_targets.squeeze().numpy(), label="True")
plt.plot(test_outputs[:, -1, :].squeeze().numpy(), label="Predicted")
plt.legend()
# Save the figure
plt.savefig('rnn_sine_wave_predictions.png')

