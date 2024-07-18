import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class EEGTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        self.pos_encoder = PositionalEncoding(64, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=64, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def wavelet_transform(data, wavelet="db4", level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return np.concatenate(coeffs, axis=-1)


# Assuming X has shape (n_trials, n_channels, n_timepoints) and y contains labels
def train_helper(X, X_test, y, y_test):
    # Feature extraction
    X_wav = np.array([wavelet_transform(trial) for trial in X])
    X_wav = X_wav.reshape(X_wav.shape[0], -1)  # Flatten to 2D

    # Normalize features
    scaler = StandardScaler()
    X_wav = scaler.fit_transform(X_wav)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_wav)
    y_tensor = torch.LongTensor(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=43)
    # X_train = X_tensor
    # y_train = y_tensor

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    input_dim = X_wav.shape[1]
    n_classes = len(np.unique(y))
    model = EEGTransformer(input_dim, n_classes, dropout=0.3)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    n_epochs = 1500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
