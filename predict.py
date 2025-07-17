import torch
import numpy as np
from train_model import PatchTST, load_data
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
def load_model():
    model = PatchTST(input_dim=1, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load("patchtst_stock_model.pth"))
    model.eval()
    return model

# Predict future stock prices
def predict_future(days=10):
    # Load last known normalized close prices and fitted scaler
    close_prices, scaler = load_data()
    model = load_model()

    # Prepare the last 30 days as input for the model
    last_sequence = close_prices[-30:].reshape(1, -1, 1)  # Shape: (1, 30, 1)
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32)

    predictions = []

    for _ in range(days):
        with torch.no_grad():
            pred = model(last_sequence).item()  # Output is still in normalized form
        predictions.append(pred)

        # Append the prediction to the sequence and shift window
        next_input = torch.tensor([[[pred]]], dtype=torch.float32)
        last_sequence = torch.cat((last_sequence[:, 1:, :], next_input), dim=1)

    # Convert list of predictions to numpy array and inverse transform to actual prices
    predictions_array = np.array(predictions).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predictions_array).flatten()

    return predicted_prices

# Run prediction and display output
if __name__ == "__main__":
    future_prices = predict_future(days=10)
    print("📈 Future Predicted Stock Prices:")
    for i, price in enumerate(future_prices, start=1):
        print(f"Day {i}: ${price:.2f}")