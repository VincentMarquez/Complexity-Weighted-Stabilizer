import pandas as pd
import numpy as np
import yfinance as yf
import lzma
import matplotlib.pyplot as plt
from scipy.stats import entropy

# --- Download S&P 500 daily close prices (2000-2024) ---
df = yf.download('^GSPC', start='2000-01-01', end='2024-01-01')
prices = df['Close'].dropna().values.flatten()  # <--- Key fix: flatten to 1D

all_dates = df.index[:len(prices)]

print("Raw prices length:", len(prices))
print("First few prices:", prices[:10])

if len(prices) < 2:
    raise ValueError("Not enough data to compute returns! Check data download.")

# --- Choose analysis type: log returns (recommended) or raw prices ---
USE_LOG_RETURNS = True   # Set to False for raw price analysis

if USE_LOG_RETURNS:
    returns = np.diff(np.log(prices))
    series = returns
    dates = all_dates[1:]    # Align dates to returns length
else:
    series = prices
    dates = all_dates

print("Returns length:", len(series))
print("First few returns/prices:", series[:10])

if len(series) == 0 or len(dates) == 0:
    raise ValueError("Time series or date array is empty! Check your data.")

if np.all(series == 0):
    raise ValueError("Series is all zeros. Something is wrong with your data.")

if len(series) != len(dates):
    minlen = min(len(series), len(dates))
    print("Auto-aligning: cutting both series and dates to", minlen)
    series = series[:minlen]
    dates = dates[:minlen]

# --- Complexity (LZMA), Entropy, "Local Lyapunov" ---
def compression_complexity(arr, method='lzma'):
    arr_bytes = arr.astype(np.float64).tobytes()
    if method == 'lzma':
        compressed = lzma.compress(arr_bytes)
    else:
        raise ValueError("Only lzma supported in this script.")
    return len(compressed) / len(arr_bytes) if len(arr_bytes) > 0 else 0.0

def sliding_window_complexity(seq, w=14):
    n = len(seq)
    kappa = np.zeros(n)
    for i in range(n):
        left = max(0, i - w // 2)
        right = min(n, i + w // 2 + 1)
        window = seq[left:right]
        if window.size == 0:
            window = np.zeros(w)
        elif len(window) < w:
            window = np.pad(window, (0, w - len(window)), mode='edge')
        kappa[i] = compression_complexity(np.array(window), method='lzma')
    return kappa

def local_entropy(seq, w=14, bins=12):
    n = len(seq)
    H = np.zeros(n)
    for i in range(n):
        left = max(0, i - w // 2)
        right = min(n, i + w // 2 + 1)
        window = seq[left:right]
        if window.size == 0:
            H[i] = 0.0
            continue
        if len(window) < w:
            window = np.pad(window, (0, w - len(window)), mode='edge')
        hist, _ = np.histogram(window, bins=bins, density=True)
        hist = hist + 1e-12  # avoid log(0)
        H[i] = entropy(hist, base=2)
    return H

def local_log_return_expansion(seq, w=14):
    n = len(seq)
    L = np.zeros(n)
    for i in range(n):
        left = max(1, i - w // 2)
        right = min(n, i + w // 2 + 1)
        window = seq[left:right]
        if window.size < 2:
            L[i] = 0.0
            continue
        diffs = np.abs(np.diff(window))
        diffs[diffs==0] = 1e-12
        L[i] = np.mean(np.log(diffs))
    return L

# --- Compute all fields (window size can be tuned) ---
W = 14  # ~2 weeks for daily data
kappa = sliding_window_complexity(series, w=W)
H = local_entropy(series, w=W)
L = local_log_return_expansion(series, w=W)

# --- Plot everything ---
plt.figure(figsize=(14,9))

plt.subplot(4,1,1)
plt.plot(dates, series, label=('Log Returns' if USE_LOG_RETURNS else 'Prices'), lw=0.8)
plt.title(f"S&P 500 {'Log Returns' if USE_LOG_RETURNS else 'Close Price'} (2000–2024)")
plt.legend()

plt.subplot(4,1,2)
plt.plot(dates, kappa, label='Kappa (lzma, w=14)')
plt.ylabel('Complexity')
plt.legend()

plt.subplot(4,1,3)
plt.plot(dates, H, label='Local Entropy (w=14)')
plt.ylabel('Entropy')
plt.legend()

plt.subplot(4,1,4)
plt.plot(dates, L, label='Local Log-Diff Expansion')
plt.ylabel('Local Lyapunov')
plt.legend()

plt.tight_layout()
plt.show()

print("Kappa: mean {:.4f}, std {:.4f}".format(np.mean(kappa), np.std(kappa)))
print("Entropy: mean {:.4f}, std {:.4f}".format(np.mean(H), np.std(H)))
print("Local Lyapunov (expansion): mean {:.4f}, std {:.4f}".format(np.mean(L), np.std(L)))
