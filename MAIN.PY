import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression

# === Indicators ===
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short=12, long=26, signal=9):
    short_ema = data['Close'].ewm(span=short, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# === IPO Dates ===
IPO_DATES = {
    "AAPL": "1980-12-12",
    "TSLA": "2010-06-29",
    "MSFT": "1986-03-13",
    "AMZN": "1997-05-15"
}

# === Fetch Data ===
def fetch_stock_data(ticker, start_year, end_year):
    ipo_date = datetime.strptime(IPO_DATES[ticker], "%Y-%m-%d")
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    start_date = max(start_date, ipo_date)
    end_date = min(end_date, datetime(2025, 1, 1))

    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if df.empty:
        print(f"No data found for {ticker}.")
        return None

    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['Binary_Class'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

# === Regression Analysis ===
def regression_analysis(data):
    data = data.dropna()
    if data.empty:
        print("Insufficient data for regression.")
        return

    data = data.sort_index()
    X = np.array(data.index.map(datetime.toordinal)).reshape(-1, 1)
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    plt.scatter(data.index, y, label='Actual Prices', color='blue', s=2)
    plt.title("Actual Stock Prices")
    plt.xticks(data.index[::len(data)//10], [d.year for d in data.index[::len(data)//10]], rotation=45)

    plt.subplot(4, 1, 2)
    plt.plot(data.index, predictions, label='Predicted Prices', linestyle="--", color='red')
    plt.title("Linear Regression Prediction")
    plt.xticks(data.index[::len(data)//10], [d.year for d in data.index[::len(data)//10]], rotation=45)

    plt.subplot(4, 1, 3)
    plt.plot(data.index, data['RSI'], color="orange")
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title("RSI Indicator")
    plt.xticks(data.index[::len(data)//10], [d.year for d in data.index[::len(data)//10]], rotation=45)

    plt.subplot(4, 1, 4)
    plt.plot(data.index, data['MACD'], color="blue")
    plt.plot(data.index, data['Signal'], color="red", linestyle='--')
    plt.title("MACD Indicator")
    plt.xticks(data.index[::len(data)//10], [d.year for d in data.index[::len(data)//10]], rotation=45)

    plt.tight_layout()
    plt.show()

# === Binary Classification ===
def binary_classification(data):
    data = data.dropna()
    if data.empty:
        print("Insufficient data for classification.")
        return
    X = data[['Close', 'RSI', 'MACD', 'Signal']]
    y = data['Binary_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    labels = ['Positive Trend', 'Negative Trend']
    sizes = [np.sum(y_pred == 1), np.sum(y_pred == 0)]
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
    plt.title("Classification Distribution")
    plt.axis('equal')
    plt.show()

# === Future Price Prediction ===
def future_prediction(data, days=30):
    data = data.dropna()
    if data.empty:
        print("Insufficient data for prediction.")
        return

    data = data.sort_index()
    X = np.array(data.index.map(datetime.toordinal)).reshape(-1, 1)
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    X_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(X_future)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, y, label="Historical Prices", color="blue")
    plt.plot(future_dates, future_preds, label="Predicted Future Prices", linestyle="--", color="green")
    plt.title(f"Future Price Prediction ({days} days ahead)")
    plt.xlabel("Year")
    plt.xticks(data.index[::len(data)//10], [d.year for d in data.index[::len(data)//10]], rotation=45)
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Display Recent Data ===
def display_current_stock_data(ticker):
    data = yf.download(ticker, period="1y")
    if data.empty:
        print("No recent data found.")
        return None
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'] = calculate_macd(data)
    print("\nLatest Data:")
    print(data[['Open', 'Close', 'Volume', 'RSI', 'MACD', 'Signal']].tail(10))
    return data

# === Main Menu ===
def main():
    ticker_map = {
        "1": ("AAPL", "APPLE"),
        "2": ("TSLA", "TESLA"),
        "3": ("MSFT", "MICROSOFT"),
        "4": ("AMZN", "AMAZON")
    }

    print("Available Companies:")
    for k, (ticker, name) in ticker_map.items():
        print(f"{k}. {ticker} ({name})")

    choice = input("Select company (1-4): ").strip()
    if choice not in ticker_map:
        print("Invalid choice.")
        return

    selected_ticker, company_name = ticker_map[choice]
    ipo_year = int(IPO_DATES[selected_ticker][:4])

    try:
        print(f"\n{company_name} IPO was in {ipo_year}. You can select data between {ipo_year} and 2025.")
        start_year = int(input(f"Enter START year (>= {ipo_year}): "))
        end_year = int(input(f"Enter END year (<= 2025): "))
        if start_year < ipo_year or end_year > 2025 or start_year > end_year:
            raise ValueError
    except ValueError:
        print("Invalid year range.")
        return

    stock_data = fetch_stock_data(selected_ticker, start_year, end_year)
    if stock_data is None:
        return

    while True:
        print(f"\nSelected Stock: {selected_ticker} ({company_name})")
        print("1. Display Historical Data Summary")
        print("2. Perform Regression Analysis")
        print("3. Perform Binary Classification")
        print("4. Display Current Stock Data")
        print("5. Predict Future Prices")
        print("6. Exit")

        option = input("Choose an option (1-6): ")

        if option == "1":
            print("Data Summary:")
            print(stock_data.head())
            print(stock_data.tail())

        elif option == "2":
            regression_analysis(stock_data)

        elif option == "3":
            binary_classification(stock_data)

        elif option == "4":
            stock_data = display_current_stock_data(selected_ticker)

        elif option == "5":
            future_prediction(stock_data)

        elif option == "6":
            print("Exiting the tool.")
            break

        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
