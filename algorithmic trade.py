import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yfinance as yf
# Define the stock symbol and time range
symbol = 'AAPL'
data = yf.download(symbol, start="2020-01-01", end="2023-01-01")
# Create features - Moving Averages, RSI, etc.
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['Price_Change'] = data['Close'].pct_change()
data['Daily_Return'] = data['Price_Change'].shift(-1)  # Shift to use as label
# Drop missing values due to rolling operations
data.dropna(inplace=True)
# Define the target variable
data['Target'] = (data['Daily_Return'] > 0).astype(int)  # Binary labels for up/down trend
# Define features and target
features = ['SMA_20', 'SMA_50', 'Price_Change']
X = data[features]
y = data['Target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
# Simulate trading based on model predictions
data['Prediction'] = model.predict(X)

# Backtest strategy
initial_balance = 10000  # Starting capital
balance = initial_balance
holdings = 0

for i in range(1, len(data)):
    if data['Prediction'].iloc[i] == 1:  # Buy Signal
        # Use .item() to get the scalar value from the Series
        if balance > data['Close'].iloc[i].item():  # Buy if we have enough balance
            holdings += 1
            balance -= data['Close'].iloc[i].item()
    elif data['Prediction'].iloc[i] == 0 and holdings > 0:  # Sell Signal
        balance += data['Close'].iloc[i].item() * holdings
        holdings = 0

final_balance = balance + holdings * data['Close'].iloc[-1]
print(final_balance)
