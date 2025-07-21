import numpy as np
import pymatching
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt
import Grok4_QEC4 as gqec

# Fix UnionFind decoder to return correct size
_original_decode = gqec.UnionFindDecoder.decode
def fixed_decode(self, syndrome):
    correction = _original_decode(self, syndrome)
    expected_size = self.n_stabilizers + 1
    if len(correction) < expected_size:
        correction = list(correction) + [0] * (expected_size - len(correction))
    return correction
gqec.UnionFindDecoder.decode = fixed_decode
print("✅ UnionFind decoder fixed!")

# Simple Surface Code
class SurfaceCode:
    def __init__(self, d):
        self.d = d
        self.n_qubits = d * d
        self.n_stabilizers = d * d - 1
        self.H = np.zeros((self.n_stabilizers, self.n_qubits), dtype=np.uint8)
        for i in range(self.n_stabilizers):
            self.H[i, i] = 1
            if (i + self.d) < self.n_qubits:
                self.H[i, (i + self.d)] = 1
    
    def measure_syndrome(self, errors):
        return (self.H @ errors) % 2
    
    def has_logical_error(self, errors):
        grid = errors.reshape((self.d, self.d))
        for col in range(self.d):
            if np.sum(grid[:, col]) == self.d:
                return True
        return False

# Test function
def test_decoders(d=5, n_trials=300):
    print(f"\nTesting d={d} with {n_trials} trials...")
    
    code = SurfaceCode(d)
    pymatch = pymatching.Matching(code.H)
    unionfind = gqec.UnionFindDecoder(code.n_stabilizers)
    
    # Create Qiskit noise
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.001, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
    readout = [[0.995, 0.005], [0.005, 0.995]]
    noise_model.add_all_qubit_readout_error(readout)
    backend = AerSimulator(noise_model=noise_model)
    
    pymatch_success = 0
    uf_success = 0
    
    for trial in range(n_trials):
        # Generate errors
        qc = QuantumCircuit(code.n_qubits)
        for i in range(code.n_qubits):
            qc.h(i)
        qc.measure_all()
        
        result = backend.run(qc, shots=1).result()
        counts = result.get_counts()
        bitstring = list(counts.keys())[0]
        errors = np.array([int(bit) for bit in reversed(bitstring)], dtype=np.uint8)
        
        syndrome = code.measure_syndrome(errors)
        
        # PyMatching
        pymatch_corr = pymatch.decode(syndrome)
        residual = (errors + pymatch_corr) % 2
        if not code.has_logical_error(residual):
            pymatch_success += 1
        
        # UnionFind
        uf_corr = unionfind.decode(syndrome.tolist())
        uf_corr = np.array(uf_corr[:code.n_qubits], dtype=np.uint8)
        residual = (errors + uf_corr) % 2
        if not code.has_logical_error(residual):
            uf_success += 1
    
    pymatch_rate = pymatch_success / n_trials * 100
    uf_rate = uf_success / n_trials * 100
    
    print(f"PyMatching: {pymatch_rate:.1f}% success")
    print(f"UnionFind:  {uf_rate:.1f}% success")
    
    if uf_rate > pymatch_rate:
        improvement = (uf_rate - pymatch_rate) / pymatch_rate * 100
        print(f"✅ UnionFind is {improvement:.1f}% better!")
    
    return pymatch_rate, uf_rate

# Main
def main():
    print("PYMATCH VS UNIONFIND - SIMPLE TEST")
    print("="*40)
    
    results = {}
    for d in [3, 5, 7]:
        pymatch_rate, uf_rate = test_decoders(d=d, n_trials=300)
        results[d] = {'PyMatching': pymatch_rate, 'UnionFind': uf_rate}
    
    # Plot
    plt.figure(figsize=(8, 6))
    distances = list(results.keys())
    pymatch_rates = [results[d]['PyMatching'] for d in distances]
    uf_rates = [results[d]['UnionFind'] for d in distances]
    
    plt.plot(distances, pymatch_rates, 'o-', label='PyMatching', markersize=10, linewidth=2)
    plt.plot(distances, uf_rates, 's-', label='UnionFind', markersize=10, linewidth=2)
    
    plt.xlabel('Code Distance d')
    plt.ylabel('Success Rate (%)')
    plt.title('PyMatching vs UnionFind')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.savefig('pymatch_vs_unionfind.png')
    plt.show()
    
    # Summary
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    avg_pymatch = np.mean(pymatch_rates)
    avg_uf = np.mean(uf_rates)
    print(f"PyMatching average: {avg_pymatch:.1f}%")
    print(f"UnionFind average:  {avg_uf:.1f}%")
    
    if avg_uf > avg_pymatch:
        print(f"\n🏆 UnionFind wins by {(avg_uf - avg_pymatch) / avg_pymatch * 100:.1f}%!")
    else:
        print(f"\n🏆 PyMatching wins!")

if __name__ == "__main__":
    main()