import pymatching
import numpy as np
import zlib

# --- 1. Setup: Small Surface Code (distance 3) ---
matching = pymatching.Matching("surface_code:distance=3")
n_qubits = matching.num_qubits
n_stabilizers = matching.num_checks

print(f"Surface code: {n_qubits} qubits, {n_stabilizers} stabilizers")

# --- 2. Complexity Proxy ---
def kolmogorov_complexity_approx(error_vec):
    error_bytes = error_vec.astype(np.uint8).tobytes()
    return len(zlib.compress(error_bytes)) / len(error_bytes)

def complexity_weighted_syndrome(syndrome, error_vec, beta=2.0):
    K = kolmogorov_complexity_approx(error_vec)
    weight = np.exp(K / beta)
    return syndrome * weight

# --- 3. Benchmark Loop ---
n_trials = 1000
p = 0.08  # physical error rate (adjust as needed)

std_detected = 0
cw_detected = 0
std_logical_errors = 0
cw_logical_errors = 0

for trial in range(n_trials):
    # Simulate random Pauli X errors (for surface code, that's enough for demo)
    error = (np.random.rand(n_qubits) < p).astype(np.int8)

    # --- Standard decoding ---
    syndrome = matching.generate_syndrome(error)
    correction = matching.decode(syndrome)
    total = (error + correction) % 2
    logical = matching.logical_commutes(total)
    std_logical_errors += int(logical)
    if np.sum(np.abs(syndrome)) > 0:
        std_detected += 1

    # --- Complexity-weighted syndrome: same syndrome, but "weight" each detection ---
    cws = complexity_weighted_syndrome(syndrome, error)
    # For detection, threshold: if any weighted syndrome is above 1.0, say detected
    if np.sum(np.abs(cws) > 1.0) > 0:
        cw_detected += 1

    # (Optional) Use the weight to "bias" the decoder (advanced—needs custom decoder)

print("\n=== PyMatching Benchmark ===")
print(f"Trials: {n_trials}, Error rate: {p}")
print(f"Standard PyMatching LER: {std_logical_errors/n_trials:.4f}")
print(f"Standard detection rate: {std_detected/n_trials:.4f}")
print(f"Complexity-weighted detection rate: {cw_detected/n_trials:.4f}")
