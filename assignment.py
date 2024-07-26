import pandas as pd
import mplfinance as mpf
import yfinance as yf

start_date = "2024-01-01"
end_date = "2024-04-30"
aapl = yf.Ticker("aapl")
symbol = "AAPL"

df = aapl.history(start=start_date, end = end_date)

df = df[["Open","High","Low","Close","Volume"]]
df.index.name = "Date"
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

print(df.head)

highs = []
lows = []
total = len(df)

for stick in range(0,total-1):
    curr_high = df["High"].iloc[stick]
    curr_low = df["Low"].iloc[stick]
    if stick > 0:
        if prev_high > curr_high:
            prev_high = curr_high
        else:
            if stick < total - 2:
                if df["High"].iloc[stick+1] < curr_high:
                    highs.append(stick)
                prev_high = curr_high
            else:
                if df["High"].iloc[stick+1] < curr_high:
                    highs.append(stick)   
        if prev_low < curr_low:
            prev_low = curr_low
        else:
            if stick < total - 2:
                if df["Low"].iloc[stick+1] > curr_low:
                    lows.append(stick)
                prev_low = curr_low
            else:
                if df["Low"].iloc[stick+1] > curr_low:
                    lows.append(stick) 
    else:
        prev_high = curr_high
        prev_low = curr_low

print(highs)
print(lows)

mpf.plot(df, type="candle",title = f"Candlestick Chart for {symbol}",ylabel="Price",style="yahoo")

supp_cands = []
res_cands = []

#mpf.show()