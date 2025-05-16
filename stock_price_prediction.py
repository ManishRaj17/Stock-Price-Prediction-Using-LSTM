import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import yfinance as yf

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Data Collection
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].values.reshape(-1, 1), stock_data.index

# 2. Feature Engineering
def create_features(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])
    
    return np.array(X), np.array(y), scaler

# 3. LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# 4. Training Function
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i + batch_size].to(device)
            batch_y = y_train_tensor[i:i + batch_size].to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 5. Prediction and Visualization
def predict_and_visualize(model, X, scaler, dates, actual_prices, seq_length):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X)):
            input_seq = torch.FloatTensor(X[i:i+1]).to(device)
            pred = model(input_seq)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(actual_prices[seq_length:])
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates[seq_length:], actual_prices, label='Actual Price', color='blue')
    plt.plot(dates[seq_length:], predictions, label='Predicted Price', color='red', linestyle='--')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('stock_prediction.png')
    plt.close()
    
    return predictions

# Main execution
if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    seq_length = 10
    train_split = 0.8
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fetch and prepare data
    prices, dates = fetch_stock_data(ticker, start_date, end_date)
    X, y, scaler = create_features(prices, seq_length)
    
    # Split data
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train model
    model = StockLSTM().to(device)
    train_model(model, X_train, y_train)
    
    # Predict and visualize
    predictions = predict_and_visualize(model, X_test, scaler, dates, prices, seq_length)
    
    # Calculate RMSE
    actual_test = scaler.inverse_transform(y_test)
    rmse = np.sqrt(np.mean((predictions - actual_test)**2))
    print(f'Root Mean Square Error: {rmse:.2f}')