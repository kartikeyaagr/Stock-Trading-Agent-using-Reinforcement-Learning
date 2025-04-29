import yfinance as yf

reliance = yf.Ticker("RELIANCE.NS")
stock_data = reliance.history(period="2y", interval="1d", actions=False)

stock_data.to_csv("reliance_OHLCV.csv")
