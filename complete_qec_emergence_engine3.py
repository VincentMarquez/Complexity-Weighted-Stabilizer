import numpy as np
import zlib
import lzma
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- Enhanced Complexity Approximator ---
def compression_complexity(arr, method='zlib'):
    arr_bytes = arr.astype(np.float64).tobytes()
    if method == 'zlib':
        compressed = zlib.compress(arr_bytes)
    elif method == 'lzma':
        compressed = lzma.compress(arr_bytes)
    else:
        raise ValueError("Unknown compression method")
    return len(compressed) / len(arr_bytes)

def sliding_window_complexity(seq, w=7, method='lzma'):
    n = len(seq)
    kappa = np.zeros(n)
    for i in range(n):
        left = max(0, i - w // 2)
        right = min(n, i + w // 2 + 1)
        window = seq[left:right]
        if len(window) < w:
            window = np.pad(window, (0, w - len(window)), mode='edge')
        kappa[i] = compression_complexity(np.array(window), method)
    return kappa

# --- Logistic Map (chaotic, float64, no artificial quantization) ---
def logistic_map_seq(x0, r, N):
    xs = np.zeros(N, dtype=np.float64)
    xs[0] = x0
    for i in range(1, N):
        xs[i] = r * xs[i-1] * (1 - xs[i-1])
    return xs

N = 4000
log_seq = logistic_map_seq(0.324932, 4.0, N)
x_idx = np.arange(N)

# --- Lyapunov Exponent (local, finite-difference, 7pt window) ---
def local_lyapunov(seq, w=7):
    # Local average of log|f'(x)| for logistic map: f'(x) = r(1-2x)
    n = len(seq)
    local_exp = np.zeros(n)
    for i in range(n):
        left = max(0, i - w // 2)
        right = min(n, i + w // 2 + 1)
        window = seq[left:right]
        derivs = np.abs(4.0 * (1 - 2 * window))
        derivs[derivs == 0] = 1e-12
        local_exp[i] = np.mean(np.log(derivs))
    return local_exp

# --- Local Entropy Field (histogram method) ---
def local_entropy(seq, w=7, bins=10):
    from scipy.stats import entropy
    n = len(seq)
    H = np.zeros(n)
    for i in range(n):
        left = max(0, i - w // 2)
        right = min(n, i + w // 2 + 1)
        window = seq[left:right]
        hist, _ = np.histogram(window, bins=bins, density=True)
        hist = hist + 1e-12  # avoid log(0)
        H[i] = entropy(hist, base=2)
    return H

# --- PSD (Global, for context) ---
f, Pxx = welch(log_seq, nperseg=256)

# --- Compute all fields (use big window, LZMA) ---
W = 7
kappa_zlib = sliding_window_complexity(log_seq, w=W, method='zlib')
kappa_lzma = sliding_window_complexity(log_seq, w=W, method='lzma')
lyap = local_lyapunov(log_seq, w=W)
H = local_entropy(log_seq, w=W, bins=14)

# --- Plot everything ---
plt.figure(figsize=(14,8))
plt.subplot(4,1,1)
plt.plot(x_idx, log_seq, lw=0.8)
plt.title('Chaotic Logistic Map')

plt.subplot(4,1,2)
plt.plot(x_idx, kappa_zlib, label='Kappa (zlib, w=7)')
plt.plot(x_idx, kappa_lzma, label='Kappa (lzma, w=7)')
plt.legend(); plt.ylabel('Complexity')

plt.subplot(4,1,3)
plt.plot(x_idx, lyap, label='Local Lyapunov (w=7)')
plt.plot(x_idx, H, label='Local Entropy (w=7)')
plt.legend(); plt.ylabel('Lyap / Entropy')

plt.subplot(4,1,4)
plt.semilogy(f, Pxx)
plt.title('Power Spectral Density (Welch)')
plt.tight_layout(); plt.show()

print("Kappa (lzma): mean {:.4f}, std {:.4f}".format(np.mean(kappa_lzma), np.std(kappa_lzma)))
print("Local Lyapunov: mean {:.4f}, std {:.4f}".format(np.mean(lyap), np.std(lyap)))
print("Local Entropy: mean {:.4f}, std {:.4f}".format(np.mean(H), np.std(H)))
