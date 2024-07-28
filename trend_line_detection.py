# bonus task
# To use deep learning to detect trend lines, you can first gether historical data 
# and standardize it. From there, any features can be created that will be used for the model
# Along with this, we will need to label the data for training. I think a CNN would work well
# here, where we train the model and use a loss function such as MSE.
# Finally, we can use an evaluation metric to evaluate our model, creating ideal predictions.


import pandas as pd
import mplfinance as mpf
import yfinance as yf
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages

# Fetches historical stock data for a given symbol and date range
def fetch_data(symbol, start_date, end_date):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

# Identifies local highs and lows in the stock data
def local_extrema(df):
    highs = []
    lows = []
    total = len(df)

    for stick in range(0, total-1):
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
    return highs, lows

# Creates the list of valid support candidates
def supp_cands(df, lows):
    supps_stats = [] # holds length, price touches (pts), slope, and weight (function of length and pts)
    supps_list = [] # holds coordinate pairs

    for i in range(0, len(lows)-1): # i = candle index with a local min
        start = df["Low"].iloc[lows[i]] # y-value at candle i
        if len(lows) == 1:
            exit
        for k in lows[i+1:]: # k = next local min
            end = df["Low"].iloc[k] # y-value at candle k
            m = (end - start) / (k - lows[i]) # slope
            length = math.sqrt((k - lows[i])**2 + (end - start)**2)
            pts = 0
            
            # checking for validity and price touches
            for j in range(lows[i]+1, k):
                if (start + (m*(j-lows[i]))) < df["Close"].iloc[j]:
                    if (start + (m*(j-lows[i]))) >= df["Low"].iloc[j]:
                        pts += 1
                else:
                    break
                # candlestick is valid
                if j == k-1:
                    weight = (length * 0.2) + (pts * 0.8)  # Adjust weights as necessary
                    supps_stats.append((length, pts, m, weight))
                    supps_list.append([(df.index[lows[i]], start), (df.index[k], end)])
    return supps_list, supps_stats

# Creates the list of valid resistance candidates
def res_cands(df, highs):
    res_stats = []
    res_list = []

    for i in range(0, len(highs)-1):
        start = df["High"].iloc[highs[i]]
        if len(highs) == 1:
            exit
        for k in highs[i+1:]:
            end = df["High"].iloc[k]
            m = (end - start) / (k - highs[i])
            length = math.sqrt((k - highs[i])**2 + (end - start)**2)
            pts = 0
            for j in range(highs[i]+1, k):
                if (start + (m*(j-highs[i]))) > df["Close"].iloc[j]:
                    if (start + (m*(j-highs[i]))) <= df["High"].iloc[j]:
                        pts += 1
                else:
                    break
                if j == k-1:
                    weight = (length * 0.2) + (pts * 0.8)  # Adjust weights as necessary
                    res_stats.append((length, pts, m, weight))
                    res_list.append([(df.index[highs[i]], start), (df.index[k], end)])
    return res_list, res_stats

# Removes any overlapping candidates, prioritizing those with the highest weight
# If there is an overlap, the candidate is only retained if there are either more price touches or longer length
def remove_overlapping_candidates(candidates_list, candidates_stats):
    non_overlapping_list = []
    non_overlapping_stats = []

    while candidates_list:
        # Determine the candidate with the most weight
        max_weight = max(candidates_stats, key=lambda x: x[3])[3]
        max_candidates = [i for i, stat in enumerate(candidates_stats) if stat[3] == max_weight]
        
        # If there are multiple candidates with the same weight, choose based on secondary criteria
        if len(max_candidates) > 1:
            max_index = max_candidates[0]
            for idx in max_candidates:
                if candidates_stats[idx][0] > candidates_stats[max_index][0] or candidates_stats[idx][1] > candidates_stats[max_index][1]:
                    max_index = idx
        else:
            max_index = max_candidates[0]

        max_candidate = candidates_list[max_index]
        max_stat = candidates_stats[max_index]

        non_overlapping_list.append(max_candidate)
        non_overlapping_stats.append(max_stat)

        # Remove the candidate and any overlapping ones
        start1, end1 = max_candidate[0][0], max_candidate[1][0]
        new_candidates_list = []
        new_candidates_stats = []

        for i in range(len(candidates_list)):
            start2, end2 = candidates_list[i][0][0], candidates_list[i][1][0]

            # Check for overlap
            if end1 < start2 or start1 > end2:
                new_candidates_list.append(candidates_list[i])
                new_candidates_stats.append(candidates_stats[i])
            else:
                # If the current candidate has higher length or more price touches, retain it
                if candidates_stats[i][0] > max_stat[0] or candidates_stats[i][1] > max_stat[1]:
                    new_candidates_list.append(candidates_list[i])
                    new_candidates_stats.append(candidates_stats[i])

        candidates_list = new_candidates_list
        candidates_stats = new_candidates_stats

    return non_overlapping_list, non_overlapping_stats

# Animates the trend lines and saves each frame to a PDF file
def animate_and_save_to_pdf(df, supps_list, res_list):
    fig, ax = plt.subplots(figsize=(8, 6))
    pdf_pages = PdfPages('trend_lines.pdf')

    def animate(i):
        ax.clear()
        if i < len(supps_list):
            mpf.plot(df, alines=dict(alines=supps_list[i], colors=['blue']), type='candle', ax=ax, ylabel="Price", style="yahoo")
        else:
            res_index = i - len(supps_list)
            mpf.plot(df, alines=dict(alines=res_list[res_index], colors=['black']), type='candle', ax=ax, ylabel="Price", style="yahoo")

        pdf_pages.savefig(fig)

    ani = animation.FuncAnimation(fig, animate, frames=len(supps_list) + len(res_list), repeat=False, interval=800)
    plt.show()
    pdf_pages.close()

# Main function to orchestrate the data fetching, processing, and plotting
def main():
    start_date = "2024-02-01"
    end_date = "2024-08-01"
    symbol = "AAPL"

    df = fetch_data(symbol, start_date, end_date)
    highs, lows = local_extrema(df)

    supps_list, supps_stats = supp_cands(df, lows)
    res_list, res_stats = res_cands(df, highs)

    non_overlapping_supps_list, non_overlapping_supps_stats = remove_overlapping_candidates(supps_list, supps_stats)
    non_overlapping_res_list, non_overlapping_res_stats = remove_overlapping_candidates(res_list, res_stats)

    animate_and_save_to_pdf(df, non_overlapping_supps_list, non_overlapping_res_list)

if __name__ == "__main__":
    main()