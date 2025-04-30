import pandas as pd


def collect_stock_data():
    print("Loading pre-downloaded stock data...")
    DATA_FILE_PATH = "/home/kartikeya.agrawal_ug25/RL_Final/train_data.csv"
    try:
        stock_data = pd.read_csv(DATA_FILE_PATH, index_col="Date", parse_dates=True)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        print("Please run the data download script first.")
        exit()
    except Exception as e:
        print(f"Error loading data from file: {e}")
        exit()
    return stock_data


def calculate_technical_indicators(df):
    df["Returns"] = df["Close"].pct_change()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_middle"] = rolling_mean
    df["BB_upper"] = rolling_mean + (2 * rolling_std)
    df["BB_lower"] = rolling_mean - (2 * rolling_std)

    df["Volatility"] = df["Returns"].rolling(window=20).std()

    return df
