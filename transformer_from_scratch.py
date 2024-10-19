import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Custom Transformer Model from scratch
class TransformerFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_heads=4, num_layers=2, dropout=0.1):
        super(TransformerFromScratch, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Positional encoding to add position information to the input sequence
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, n_heads, hidden_size * 4, dropout) for _ in range(num_layers)]
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Input embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Output layer (take the last time step's hidden state for prediction)
        output = self.fc_out(x[:, -1, :])  # Predict next value in sequence
        return output

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

# Transformer Encoder Layer from scratch
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_hidden_size, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadSelfAttention(hidden_size, n_heads, dropout)
        self.ff = FeedForward(hidden_size, ff_hidden_size, dropout)
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention layer
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        
        # Feed-forward layer
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm
        
        return x

# Multi-head self-attention layer from scratch
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % n_heads == 0
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        
        # Linear projections for query, key, value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Split Q, K, V into multiple heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate attention output for all heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Final linear layer
        output = self.fc_out(attn_output)
        return output

# Feed-forward network for the transformer
class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

# Step 2: Train the custom Transformer
input_size = 1  # Each input is a scalar (1 feature)
hidden_size = 64  # Hidden size for the Transformer
output_size = 1  # Output is a scalar (next sine value)
batch_size = 32
num_epochs = 100

# Instantiate the model
model = TransformerFromScratch(input_size, hidden_size, output_size)
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
    test_loss = criterion(test_outputs, test_targets)
    print(f'Test Loss: {test_loss.item()}')
    t = torch.zeros([198])
    test_loss = criterion(t, test_targets.squeeze())
    print(f'Zero Loss: {test_loss.item()}')

# Step 4: Plot the results
plt.plot(test_targets.squeeze().numpy(), label="True")
plt.plot(test_outputs.numpy(), label="Predicted")
plt.legend()
# Save the figure
plt.savefig('transformer_sine_wave_predictions.png')
