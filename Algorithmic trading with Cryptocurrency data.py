import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk, messagebox


def display_results(signals):
    # Plotting
    plt.figure(figsize=(10, 6))

    # Price and Moving Average
    plt.plot(signals['Price'], label='Crypto Price', alpha=0.5)
    plt.plot(signals['MA'], label=f'{50}-day Moving Average', linestyle='--', alpha=0.5)

    # Buy signals
    plt.scatter(signals.index[signals['Buy_Signal'] == 1], signals['Price'][signals['Buy_Signal'] == 1], marker='^',
                color='g', label='Buy Signal')

    # Sell signals
    plt.scatter(signals.index[signals['Sell_Signal'] == -1], signals['Price'][signals['Sell_Signal'] == -1],
                marker='v', color='r', label='Sell Signal')

    # Displaying the plot
    plt.title('Crypto Algorithmic Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Crypto Price')
    plt.legend()
    plt.show()


class CryptoTradingApp:
    def __init__(self, boot):
        self.boot = boot
        self.boot.title("Crypto Trading Algorithm")

        self.label_ticker = ttk.Label(boot, text="Enter Crypto Ticker:")
        self.label_ticker.grid(row=0, column=0, padx=10, pady=10)

        self.entry_ticker = ttk.Entry(boot)
        self.entry_ticker.grid(row=0, column=1, padx=10, pady=10)

        self.label_start_date = ttk.Label(boot, text="Enter Start Date (YYYY-MM-DD):")
        self.label_start_date.grid(row=1, column=0, padx=10, pady=10)

        self.entry_start_date = ttk.Entry(boot)
        self.entry_start_date.grid(row=1, column=1, padx=10, pady=10)

        self.label_end_date = ttk.Label(boot, text="Enter End Date (YYYY-MM-DD):")
        self.label_end_date.grid(row=2, column=0, padx=10, pady=10)

        self.entry_end_date = ttk.Entry(boot)
        self.entry_end_date.grid(row=2, column=1, padx=10, pady=10)

        self.label_predict_days = ttk.Label(boot, text="Enter the number of days to predict:")
        self.label_predict_days.grid(row=3, column=0, padx=10, pady=10)

        self.entry_predict_days = ttk.Entry(boot)
        self.entry_predict_days.grid(row=3, column=1, padx=10, pady=10)

        self.button_run_strategy = ttk.Button(boot, text="Run Strategy", command=self.run_strategy)
        self.button_run_strategy.grid(row=4, column=0, columnspan=2, pady=10)

    def run_strategy(self):
        # Get input from the user
        ticker = self.entry_ticker.get()
        start_date = self.entry_start_date.get()
        end_date = self.entry_end_date.get()

        # Fetch data
        data = get_crypto_data(ticker, start_date, end_date)

        # Define strategy parameters
        ma_window = 50
        rsi_window = 14
        overbought_threshold = 70
        oversold_threshold = 30

        # Generate signals
        signals = generate_signals(data, ma_window, rsi_window, overbought_threshold, oversold_threshold)

        # Backtest the strategy
        signals = backtest_strategy(signals)

        # Display results
        display_results(signals)

        # Get the number of days to predict from the user
        n_days_to_predict = int(self.entry_predict_days.get())

        # Train the machine learning model
        ml_model, accuracy = train_ml_model(data, n_days=n_days_to_predict)

        # Make a prediction for the next n_days
        last_data = data.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1)
        prediction = ml_model.predict(last_data)

        accuracy = accuracy*100

        # Display the accuracy, and predicted trend in a single dialog box
        message = f"Model Accuracy: {accuracy:.2f}%\n"
        message += f"The predicted trend after {n_days_to_predict} days is: {'Uptrend' if prediction[0] == 1 else 'Downtrend'}"

        messagebox.showinfo("Model Accuracy and Prediction", message)


def get_crypto_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def calculate_ma(data, window):
    return data['Close'].rolling(window=window).mean()


def calculate_rsi(data, window):
    diff = data['Close'].diff()
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(data, ma_window, rsi_window, overbought_threshold, oversold_threshold):
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']

    signals['MA'] = calculate_ma(data, ma_window)
    signals['RSI'] = calculate_rsi(data, rsi_window)

    signals['Buy_Signal'] = np.where((signals['MA'] > signals['Price']) & (signals['RSI'] < oversold_threshold), 1, 0)
    signals['Sell_Signal'] = np.where((signals['MA'] < signals['Price']) & (signals['RSI'] > overbought_threshold), -1,
                                      0)

    return signals


def backtest_strategy(signals):
    signals['Position'] = signals['Buy_Signal'] + signals['Sell_Signal']
    signals['Position'] = signals['Position'].cumsum()

    return signals


def train_ml_model(data, target_col='Target', n_days=5):
    data[target_col] = np.where(data['Close'].shift(-n_days) > data['Close'], 1, 0)

    x = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


root = tk.Tk()
app = CryptoTradingApp(root)
root.mainloop()
