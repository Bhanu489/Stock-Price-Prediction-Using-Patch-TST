import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# === PatchTST Model (Simplified) ===
class PatchTST(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PatchTST, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# === Custom Dataset for Time Series ===
class StockDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.data[index+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === Data Preprocessing ===
def load_data(file="processed_stock_data.csv"):
    df = pd.read_csv(file)
    close_prices = df["Close"].values

    # Normalize data
    scaler = MinMaxScaler()
    close_prices = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

    return close_prices, scaler

# === Training Function ===
def train():
    # Load and prepare data
    close_prices, scaler = load_data()
    dataset = StockDataset(close_prices)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = PatchTST(input_dim=1, hidden_dim=64, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(20):  # 20 epochs
        for x, y in dataloader:
            x = x.unsqueeze(-1)  # Add feature dimension
            y = y.unsqueeze(-1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.5f}")

    # Save model
    torch.save(model.state_dict(), "patchtst_stock_model.pth")
    print(" Model trained and saved as 'patchtst_stock_model.pth'")

if __name__ == "__main__":
    train()

