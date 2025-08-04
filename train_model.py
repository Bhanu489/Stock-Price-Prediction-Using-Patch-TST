# train_model.py -- Canary Test Version

import torch
from torch import nn
from preprocess import create_patches
import pandas as pd
import numpy as np

class PatchTST(nn.Module):
    def __init__(self, patch_len, pred_len, d_model):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # --- THIS IS THE CANARY TEST ---
        # If you don't see this message in the error, the old code is running.
        print("\n\n>>> EXECUTING THE NEW, CORRECT FORWARD METHOD <<<\n")
        print(f">>> Initial shape of tensor x: {x.shape}\n")
        
        # This is the corrected logic.
        x = x.squeeze(-1)
        x = x.unsqueeze(1)
        
        print(f">>> Final shape before linear layer: {x.shape}\n")
        
        # This is the line that was crashing.
        x = self.patch_embedding(x)
        
        x = self.encoder(x)
        out = self.head(x[:, 0, :])
        return out

if __name__ == "__main__":
    # The rest of the file is the same...
    try:
        df = pd.read_csv("data/AAPL.csv")
    except FileNotFoundError:
        print("Error: 'data/AAPL.csv' not found. Run fetch_data.py first.")
        exit()

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    prices = df["Close"].values

    price_mean = prices.mean()
    price_std = prices.std()
    prices = (prices - price_mean) / price_std

    patch_len = 16
    pred_len = 7
    x, y = create_patches(prices, patch_len, pred_len)

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32)

    model = PatchTST(patch_len, pred_len, d_model=64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting model training...")
    for epoch in range(50):
        model.train()
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item()}")
    print("Training complete.")

    torch.save(model.state_dict(), "patchtst_stock_model.pth")
    print("Model saved to patchtst_stock_model.pth")