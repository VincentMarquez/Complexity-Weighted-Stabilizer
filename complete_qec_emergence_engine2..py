import numpy as np
import zlib
import matplotlib.pyplot as plt

# --- Utility: Kolmogorov Complexity Approximation ---
def kolmogorov_complexity_approx(arr: np.ndarray) -> float:
    """
    Approximate Kolmogorov complexity by zlib compression ratio.
    arr: np.ndarray of floats (small window, e.g. 3 elements)
    """
    arr_bytes = arr.tobytes()
    compressed = zlib.compress(arr_bytes)
    return len(compressed) / len(arr_bytes)

# --- Derivative Operators ---
def standard_derivative(f, x, dx=1e-4):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def analytic_weighted_derivative(f, x, dx=1e-4, weight=lambda x: 1+np.sin(x)):
    return weight(x) * standard_derivative(f, x, dx)

def complexity_weighted_derivative(f, x, dx=1e-4):
    window = np.array([f(x - dx), f(x), f(x + dx)])
    K = kolmogorov_complexity_approx(window)
    return K * standard_derivative(f, x, dx)

def helical_derivative(f, x, dx=1e-4):
    v = standard_derivative(f, x, dx)
    fpp = (f(x + dx) - 2*f(x) + f(x - dx)) / dx**2
    theta = np.arctan2(fpp, v) if v != 0 else 0.0
    window = np.array([f(x - dx), f(x), f(x + dx)])
    kappa = kolmogorov_complexity_approx(window)
    return np.array([v, theta, kappa])

# --- Test Functions: Sine and Logistic Map ---
def f_sine(x):
    return np.sin(x)

def logistic_map_seq(x0, r, N):
    xs = [x0]
    for _ in range(N-1):
        xs.append(r * xs[-1] * (1 - xs[-1]))
    return np.array(xs)

# --- Compute on Sine Wave ---
x_vals = np.linspace(0, 10, 400)
sine_vals = np.sin(x_vals)
std_deriv = np.gradient(sine_vals, x_vals)
analytic_mod = [analytic_weighted_derivative(f_sine, x) for x in x_vals]
complexity_mod = [complexity_weighted_derivative(f_sine, x) for x in x_vals]
vecs_sine = np.array([helical_derivative(f_sine, x) for x in x_vals])

plt.figure(figsize=(12,4))
plt.plot(x_vals, std_deriv, label="Standard Derivative")
plt.plot(x_vals, analytic_mod, label="Analytic-Weighted Derivative")
plt.plot(x_vals, complexity_mod, label="Complexity-Weighted Derivative")
plt.title("Derivatives for Sine Wave")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,4))
plt.plot(x_vals, vecs_sine[:,0], label="v(x): Derivative")
plt.plot(x_vals, vecs_sine[:,1], label="theta(x): Phase")
plt.plot(x_vals, vecs_sine[:,2], label="kappa(x): Complexity")
plt.title("Helical Derivative Components (Sine Wave)")
plt.legend(); plt.tight_layout(); plt.show()

print("Sine kappa: mean {:.4f}, std {:.4f}".format(
    np.mean(vecs_sine[:,2]), np.std(vecs_sine[:,2]))
)

# --- Compute on Logistic Map (Chaotic) ---
N = 400
log_seq = logistic_map_seq(0.3, 4.0, N)
x_idx = np.arange(N)

def logistic_wrapper(i):
    idx = int(np.clip(i, 0, N-1))
    return log_seq[idx]

std_deriv_log = np.gradient(log_seq, x_idx)
analytic_mod_log = [analytic_weighted_derivative(logistic_wrapper, i) for i in x_idx]
complexity_mod_log = [complexity_weighted_derivative(logistic_wrapper, i) for i in x_idx]
vecs_log = np.array([helical_derivative(logistic_wrapper, i) for i in x_idx])

plt.figure(figsize=(12,4))
plt.plot(x_idx, std_deriv_log, label="Standard Derivative")
plt.plot(x_idx, analytic_mod_log, label="Analytic-Weighted Derivative")
plt.plot(x_idx, complexity_mod_log, label="Complexity-Weighted Derivative")
plt.title("Derivatives for Logistic Map (Chaotic)")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,4))
plt.plot(x_idx, vecs_log[:,0], label="v(x): Derivative")
plt.plot(x_idx, vecs_log[:,1], label="theta(x): Phase")
plt.plot(x_idx, vecs_log[:,2], label="kappa(x): Complexity")
plt.title("Helical Derivative Components (Logistic Map)")
plt.legend(); plt.tight_layout(); plt.show()

print("Logistic map kappa: mean {:.4f}, std {:.4f}".format(
    np.mean(vecs_log[:,2]), np.std(vecs_log[:,2]))
)
