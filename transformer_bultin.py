import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Positional Encoding class to add positional information to input embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, hidden_size)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register as a buffer (non-learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Custom Transformer model using PyTorch's built-in Transformer module
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_encoder_layers=2, n_heads=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size

        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,  # Embedding size
                nhead=n_heads,  # Number of heads in the multi-head attention
                dim_feedforward=hidden_size * 4,  # Size of feedforward network
                dropout=dropout
            ),
            num_layers=num_encoder_layers  # Number of encoder layers
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply embedding layer and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Take the last time step's output to predict the next value
        output = self.fc_out(x[:, -1, :])
        return output

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

# Step 2: Train the built-in Transformer model
input_size = 1  # Each input is a scalar (1 feature)
hidden_size = 64  # Hidden size for the Transformer
output_size = 1  # Output is a scalar (next sine value)
batch_size = 32
num_epochs = 100

# Instantiate the model
model = TransformerModel(input_size, hidden_size, output_size)
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

        optimizer.zero_grad()  # Zero gradients
        outputs = model(batch_inputs)  # Forward pass
        loss = criterion(outputs, batch_targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss/len(train_inputs)}')

# Step 3: Test the model
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_targets.squeeze())
    print(test_outputs.size())
    print(f'Test Loss: {test_loss.item()}')
    t = torch.zeros([198, 1])
    test_loss = criterion(t, test_targets.squeeze())
    print(f'Zero Loss: {test_loss.item()}')

# Step 4: Plot the results
plt.plot(test_targets.squeeze().numpy(), label="True")
plt.plot(test_outputs.numpy(), label="Predicted")
plt.legend()
# Save the figure
plt.savefig('transformer_builtin_sine_wave_predictions.png')
