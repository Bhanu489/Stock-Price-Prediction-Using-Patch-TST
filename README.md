
# 📈 Stock Price Prediction with PatchTST

This project implements a full machine learning pipeline to forecast future stock prices of Apple Inc. (AAPL) using a simplified version of the **PatchTST (Patch Time Series Transformer)** model. The predictions are served via a RESTful API built using **Flask**.

---

## 🚀 Features

- **📊 Automated Data Fetching**: Downloads historical stock data using `yfinance`.
- **🔧 Data Preprocessing**: Cleans and normalizes using `MinMaxScaler`.
- **📉 Time Series Forecasting**: Forecasts prices using a PatchTST model implemented in `PyTorch`.
- **🔁 Modular Design**: Clean separation of functionality across scripts.
- **🌐 REST API**: Serve future stock predictions through a GET API endpoint.

---
## Project Architecture
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/7017b371-e817-4e3e-8624-28dd373707eb" />


## 🛠️ Tech Stack

- **Python**
- **Flask** – Web API
- **PyTorch & PyTorch Lightning** – Deep learning model
- **Pandas, NumPy** – Data processing
- **Scikit-learn** – Normalization
- **yfinance** – Fetching stock data

---

## 📁 Project Structure

```
.
├── app.py                    # Flask web server
├── fetch_data.py             # Downloads stock data
├── preprocess.py             # Cleans & normalizes data
├── train_model.py            # Trains the PatchTST model
├── predict.py                # Generates future predictions
├── AAPL.csv                  # Raw stock data
├── processed_stock_data.csv  # Cleaned & normalized data
└── patchtst_stock_model.pth  # Trained model weights
```

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Ensure `requirements.txt` contains:
```
torch
pandas
numpy
scikit-learn
yfinance
flask
pytorch-lightning
```

Then install:
```bash
pip install -r requirements.txt
```

---

## 🧪 Usage Guide

Run the scripts in the following order:

###  Step 1: Fetch Stock Data
```bash
python fetch_data.py
```

###  Step 2: Preprocess the Data
```bash
python preprocess.py
```

###  Step 3: Train the Model
```bash
python train_model.py
```

###  Step 4: Start the Flask Server
```bash
python app.py
```

App will be available at:  
**http://127.0.0.1:5000**

---

## 🔮 API Usage

**Endpoint:** `GET /predict`  
**Query Parameter:**
- `days` (optional): Number of future days to predict (default = 10)

### Example Request
```bash
curl "http://127.0.0.1:5000/predict?days=15"
```

### Example Response
```json
{
  "predictions":
  [ 0.9967188239097595, 0.9985713362693787, 1.0017839670181274, 
  1.0058801174163818, 1.0105863809585571, 1.015715479850769, 
  1.0211377143859863, 1.0267635583877563, 1.0325305461883545, 
  1.038397192955017] 
}
```

---

## 📬 Contributing

Contributions are welcome! Feel free to fork this repo and open a pull request.

---


---

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/)
- [PatchTST: Efficient Long-Term Time Series Forecasting with Patches](https://arxiv.org/abs/2211.14730)
