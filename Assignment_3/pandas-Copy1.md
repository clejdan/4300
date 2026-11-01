# Data Manipulation Cheat Sheet (pandas + numpy)

Built around the kind of data you're working with in Minance: stock price DataFrames from `yfinance`. Every example below assumes you already have a DataFrame called `data`.
```python
import pandas as pd
import numpy as np
import yfinance as yf
import torch.utils.data import Dataset, DataLoader

df = yf.download("AAPL", start="2020-01-01", end="2026-01-01")
```
---

## cool settings, side note
- pd.set_option('display.max_rows', None)

- print("Unknown" in df["country"].unique())

- .tolist()
## 1. First things to check on any new DataFrame

```python
df.head()        # first 5 rows, quick sanity check
df.tail()         # last 5 rows
df.shape          # (rows, columns), e.g. (1500, 6)
df.columns        # list of column names
df.dtypes         # data type of each column
df.info()         # combines shape, dtypes, and null counts
df.describe()     # mean, std, min, max, quartiles for numeric columns
df.duplicated().sum() # check the # of duplicated rows in the dataframe
```

Run these every time you load new data. Half of debugging a broken pipeline is catching a wrong shape or an unexpected NaN early.

**Why this matters for ML:** a model trains on whatever shape and values you hand it, no error message tells you "this looks wrong" ahead of time. If your data has an unexpected NaN or the wrong number of rows, you often don't find out until loss prints `nan` mid-training. Checking shape and nulls before you build anything catches this early, when it's a 1-line fix instead of a debugging session.

---

## 2. Selecting columns

```python
df['Close']                    # one column, returns a Series
df[['Close', 'Volume']]        # multiple columns, returns a DataFrame (note the double brackets)
df["Column"].value_counts()    #get unique values alongside their counts in the dataframe
```

**One bracket = Series (1D). Two brackets = DataFrame (2D)**, even if it's just one column. This trips people up constantly, keep it in mind.

**Why this matters for ML:** PyTorch models expect 2D input, shaped as (number of samples, number of features). A Series is 1D. If you feed a Series into `torch.tensor()` without reshaping, your tensor ends up the wrong shape and your model's forward pass either crashes or silently computes something incorrect. This is the root cause of the `.view(-1, 1)` step you already use in your linear regression script.

---

## 3. Selecting rows

```python
df.iloc[0]              # first row, by position
df.iloc[0:10]           # first 10 rows, by position
df.iloc[-5:]            # last 5 rows

df.loc['2024-01-01']    # row by label (here, the date index)
df.loc['2024-01-01':'2024-01-10']   # range by label
```

`iloc` = position-based (like list indexing). `loc` = label-based (uses the actual index values, like dates).

**Why this matters for ML:** this is how you do your train/test split for time series data correctly. You already learned that shuffling stock data before splitting leaks future information into training. `iloc` with a fixed position range (like `df.iloc[:split_idx]`) is exactly how you split chronologically instead of randomly.

---

## 4. Filtering rows (boolean masks)

This is the one you'll use constantly.

```python
df[df['Close'] > 150]                     # rows where Close is above 150
df[(df['Close'] > 150) & (df['Volume'] > 1000000)]   # multiple conditions, use & not "and"
df[df['Close'].isna()]                     # rows where Close is missing
df[~df['Close'].isna()]                    # rows where Close is NOT missing (~ means "not")
```

Important: use `&` and `|`, not `and`/`or`, when combining conditions on a Series. Also wrap each condition in parentheses.

**Why this matters for ML:** filtering is how you remove bad rows before training (missing values, corrupted rows, outlier price spikes from stock splits or data errors) and how you build labels for classification tasks, like your VLM plausibility project's yes/no labels. If you use `and`/`or` instead of `&`/`|`, pandas throws an error rather than silently doing the wrong thing, so at least this mistake is loud, not silent.

---

## 5. Creating new columns

```python
df['return'] = df['Close'].pct_change()          # percent change from previous row
df['next_close'] = df['Close'].shift(-1)          # shift column UP by 1 (tomorrow's value, on today's row)
df['prev_close'] = df['Close'].shift(1)           # shift column DOWN by 1 (yesterday's value)
df['ma_5'] = df['Close'].rolling(window=5).mean() # 5-day rolling average
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))  # log returns
```

`shift(-1)` looks forward (useful for prediction targets). `shift(1)` looks backward (useful for lagged features). Getting the sign backwards here is a common, quiet bug, double check which direction you meant.

**Why this matters for ML:** this is literally how you build a supervised learning problem out of raw price data. `shift(-1)` is how you create your target column, "what should the model predict," by pulling tomorrow's price onto today's row. Rolling averages and log returns are common engineered features because raw prices are noisy and non-stationary (their statistical properties drift over time), which makes models harder to train. Log returns specifically are used across quant finance because they're more stable and closer to normally distributed than raw price changes.

---

## 6. Handling missing data (NaN)

Rolling averages and shifts always create NaN values at the start or end of your data. You must handle these before feeding data into a model.

```python
df.isna().sum()             # count of NaN values per column, always check this after feature engineering
df.dropna()                 # drop any row with at least one NaN
df.dropna(subset=['ma_5'])  # only drop rows where ma_5 specifically is NaN
df.fillna(0)                # replace NaN with 0 (rarely the right choice for price data)
df.fillna(method='ffill')   # forward-fill: carry the last valid value forward
```

For Minance specifically: after adding rolling averages or shifts, always run `data.dropna()` before converting to tensors. A NaN inside a tensor will silently corrupt your loss calculation (loss becomes `nan` and training breaks).

**Why this matters for ML:** neural networks do pure math on every value you give them. There's no built-in "skip this one" behavior. One NaN anywhere in a batch propagates through the entire forward pass and back through `.backward()`, turning your whole loss and every gradient into `nan`. Once that happens, your model weights are permanently corrupted for the rest of training, you have to restart from a clean checkpoint. This is the same class of bug you already hit with your VLM training loss logging, small oversights early in the pipeline cause confusing failures much later.

---

## 7. Sorting

```python
df.sort_values('Volume')                  # ascending by default
df.sort_values('Volume', ascending=False) # descending
df.sort_index()                            # sort by the index (usually date)
```

**Why this matters for ML:** less critical day to day, but `sort_index()` matters any time you've merged or filtered data, since those operations can scramble row order. Rolling averages and shifts assume chronological order. If your dates are out of order when you call `.rolling()` or `.shift()`, you get numbers that look valid but are computed against the wrong neighboring rows, a silent correctness bug, not a crash.

---

## 8. Grouping and aggregating

Less common for single-stock work, but essential once you pull multiple tickers.

```python
df.groupby('ticker')['Close'].mean()       # average close price per ticker
df.groupby('ticker').agg({
    'Close': 'mean',
    'Volume': 'sum'
})
```

**Why this matters for ML:** once Minance covers more than one stock, you can't compute a rolling average or normalize price across the whole dataset at once, that would blend Apple's price scale with a penny stock's price scale. You need to group by ticker first, then apply the feature engineering per group. This is the same "grouped" principle you already used correctly in your VLM project, keeping each image's 9 rows together during the train/val/test split, applied here to keep each stock's rows together during feature engineering.

---

## 9. Merging multiple DataFrames

Useful once you're combining price data with another source (fundamentals, SEC filing dates, etc).

```python
merged = pd.merge(df1, df2, on='date', how='inner')   # only keep rows where 'date' exists in both
merged = pd.merge(df1, df2, on='date', how='left')    # keep all rows from df1, fill missing df2 values with NaN
```

`how='inner'` is the safest default when you're not sure, it avoids silently introducing NaN rows.

**Why this matters for ML:** this is how you'd combine price data with a second feature source later, like SEC filing text or sentiment scores, into one row per date that your model can train on. Each row needs every feature aligned to the same date. Get the merge wrong and you can silently pair a stock's price on one date with a filing from a different date, which trains the model on a false relationship.

---

## 10. Converting to numpy / PyTorch tensors

This is the step where your pandas work hands off to your model.

```python
# pandas -> numpy
arr = df['Close'].values          # returns a numpy array

# pandas -> PyTorch tensor directly
import torch
tensor = torch.tensor(df['Close'].values, dtype=torch.float32)

# Reshape a 1D column into a (N, 1) column vector, needed for most model inputs
tensor = tensor.view(-1, 1)
```

`.values` is the standard way to strip a Series or DataFrame down to a raw numpy array. Always specify `dtype=torch.float32` explicitly, pandas sometimes infers `float64`, and PyTorch models expect `float32` by default. Mismatched dtypes cause confusing errors mid-training.

**Why this matters for ML:** this is the exact boundary between "data work" and "model work" in every script you've written for Minance and your VLM project. `.view(-1, 1)` reshapes a flat list of N prices into an (N, 1) matrix, matching the (batch_size, num_features) shape `nn.Linear` expects. Get the dtype wrong and PyTorch throws a type mismatch error the moment your data meets your model's weights, since the weights are `float32` by default and can't do matrix multiplication against `float64` data without an explicit cast.

---

## 11. Quick reference: common mistakes to watch for

| Mistake | What happens | Why it matters for ML | Fix |
|---|---|---|---|
| Shuffling time series data before train/test split | Leaks future data into training | Model looks accurate during testing but fails in real deployment, since it learned from data it should never have seen yet | Split by date order, never shuffle time series |
| Forgetting `.dropna()` after `.rolling()` or `.shift()` | NaN values crash training silently (loss becomes `nan`) | One NaN propagates through the entire forward and backward pass, corrupting every weight in the model | Always check `data.isna().sum()` after feature engineering |
| Using `and`/`or` instead of `&`/`\|` in filters | Raises a `ValueError` | Not silent, but wastes debugging time if you don't recognize the error immediately | Use `&` and `\|`, wrap each condition in parentheses |
| Single bracket `data['Close']` when you need a DataFrame | Returns a Series, breaks code expecting 2D shape | PyTorch models expect 2D tensors, a 1D Series converts to the wrong tensor shape | Use double brackets `data[['Close']]` |
| Not specifying `dtype=torch.float32` | Tensor defaults to `float64`, mismatches with model weights | Matrix multiplication between float32 weights and float64 data throws a type error | Always set `dtype=torch.float32` explicitly |
| Forgetting to normalize price data before training | Large raw values (100s) destabilize gradient descent | Large input values produce large gradients, which can cause loss to explode or oscillate instead of converging smoothly | Subtract mean, divide by std, before training |

---

## How to use this

Don't memorize this top to bottom. Keep it open in a side tab while you build Minance, and look things up as you hit them. The goal is recognizing the pattern when you need it, not reciting the syntax from memory. That recall will come naturally after you've used each one a few times in real code.