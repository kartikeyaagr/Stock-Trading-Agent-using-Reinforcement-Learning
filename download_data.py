import yfinance as yf

reliance = yf.Ticker("RELIANCE.NS")
stock_data = reliance.history(period="2y", interval="1d", actions=False)

stock_data.to_csv("reliance_OHLCV.csv")
eval_data = reliance.history(interval="1d", start="2020-05-04", end="2022-05-04")
eval_data.to_csv("eval_data.csv")
