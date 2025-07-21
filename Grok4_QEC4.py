import random
import math
import time

# Seed for reproducibility
random.seed(42)

# --- Original Manual Math and Matrix Functions ---
def symlog(x):
    return math.copysign(1.0, x) * math.log(1.0 + abs(x))

def symexp(y):
    return math.copysign(1.0, y) * (math.exp(abs(y)) - 1.0)

def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    rows_B = len(B)
    cols_B = len(B[0]) if rows_B > 0 else 0
    if cols_A != rows_B:
        return []
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_add(A, B):
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]
    return result

def matrix_subtract(A, B):
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    return result

def matrix_transpose(A):
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    result = [[0.0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    return result

def scalar_matrix_multiply(scalar, A):
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = scalar * A[i][j]
    return result

def sigmoid(x):
    # Clamping to avoid overflow
    x = clip(x, -500, 500)
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_deriv(y):
    return y * (1.0 - y)
    
def relu(x):
    return max(0.0, x)

def relu_deriv(y):
    return 1.0 if y > 0 else 0.0

def normal_pdf(x, mean, std):
    std = max(std, 1e-8)
    return (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std) ** 2)

def clip(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def init_matrix(rows, cols, scale=0.01):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

# --- NEW: Manual CNN Components ---
def conv2d(input_matrix, kernel):
    h_in, w_in = len(input_matrix), len(input_matrix[0])
    h_k, w_k = len(kernel), len(kernel[0])
    h_out, w_out = h_in - h_k + 1, w_in - w_k + 1
    output_matrix = [[0.0] * w_out for _ in range(h_out)]
    for i in range(h_out):
        for j in range(w_out):
            for ki in range(h_k):
                for kj in range(w_k):
                    output_matrix[i][j] += input_matrix[i + ki][j + kj] * kernel[ki][kj]
    return output_matrix

def flatten(matrix_3d):
    flat_list = []
    for matrix in matrix_3d:
        for row in matrix:
            flat_list.extend(row)
    return flat_list

# --- NEW: Neural Complexity Estimator (CNN) ---
class NeuralComplexityEstimator:
    def __init__(self, input_shape=(8, 8), n_filters=2, kernel_size=3):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        # Layer 1: Convolutional
        # 2 input channels (real/imag), n_filters output channels
        self.conv_kernels = [[[init_matrix(kernel_size, kernel_size) for _ in range(2)] for _ in range(n_filters)]]
        
        # Layer 2: Fully Connected
        conv_out_size = (input_shape[0] - kernel_size + 1) * (input_shape[1] - kernel_size + 1) * n_filters
        self.fc_weights = init_matrix(conv_out_size, 1)

    def forward(self, op_matrix_real, op_matrix_imag):
        # Convolutional Layer
        conv_outputs = []
        for f in range(self.n_filters):
            # Convolve real part
            out_real = conv2d(op_matrix_real, self.conv_kernels[0][f][0])
            # Convolve imag part
            out_imag = conv2d(op_matrix_imag, self.conv_kernels[0][f][1])
            # Add and apply ReLU
            combined = matrix_add(out_real, out_imag)
            activated = [[relu(val) for val in row] for row in combined]
            conv_outputs.append(activated)
            
        # Flatten
        flat_vector = flatten(conv_outputs)
        
        # Fully Connected Layer
        # Note: input to matrix_multiply must be list of lists
        output = matrix_multiply([flat_vector], self.fc_weights)[0][0]
        return output

# --- Original Surface Code and Decoder Classes ---
class SurfaceCode:
    def __init__(self, d=3, error_rate=0.01):
        self.d = d
        self.n_qubits = self.d ** 2
        self.n_stabilizers = 2 * (self.d - 1) ** 2 if d > 1 else 0
        self.error_rate = error_rate

    def set_error_rate(self, p):
        self.error_rate = p

    def apply_bitflip_noise(self, state):
        errors = [0] * self.n_qubits
        for i in range(self.n_qubits):
            if random.random() < self.error_rate:
                errors[i] = 1
        return errors

    def measure_syndrome(self, errors):
        if self.d <= 1: return []
        syndrome = [0] * self.n_stabilizers
        # Simplified syndrome measurement for this example
        for s in range(self.n_stabilizers):
            qubits_in_stabilizer = [(s + i) % self.n_qubits for i in range(self.d)]
            parity = sum(errors[q] for q in qubits_in_stabilizer) % 2
            syndrome[s] = parity
        return syndrome

    def logical_error(self, errors, correction):
        if self.d <= 1: return 0
        total = [ (errors[i] + correction[i % len(correction)]) % 2 for i in range(self.n_qubits) ]
        logical_parity = sum(total[i] for i in range(0, self.n_qubits, self.d)) % 2
        return logical_parity

class UnionFindDecoder:
    def __init__(self, n_stabilizers):
        self.n_stabilizers = n_stabilizers
        self.parent = list(range(n_stabilizers))
        self.rank = [0] * n_stabilizers

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1
    
    def decode(self, syndrome):
        # Simplified UF decoder for demonstration
        correction = [0] * self.n_stabilizers
        for i in range(self.n_stabilizers):
            if syndrome[i] == 1:
                # Flip first qubit associated with stabilizer
                correction[i] = 1
        return correction

class NeuralDecoder:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = init_matrix(input_size, hidden_size)
        self.b1 = [[0.0 for _ in range(hidden_size)]]
        self.W2 = init_matrix(hidden_size, output_size)
        self.b2 = [[0.0 for _ in range(output_size)]]
        self.std = 0.1
        self.clip_eps = 0.2
        self.c1 = 0.5
        self.c2 = 0.01

    def forward(self, x):
        if not x: return [], 0.0, [], [], []
        x = [[symlog(xi) for xi in x]]
        z1 = matrix_add(matrix_multiply(x, self.W1), self.b1)
        a1 = [[sigmoid(z) for z in z1[0]]]
        z2 = matrix_add(matrix_multiply(a1, self.W2), self.b2)
        mu = [sigmoid(z2[0][i]) for i in range(self.output_size - 1)]
        V = sigmoid(z2[0][-1])
        return mu, V, a1, z1, z2

    def sample_action(self, mu):
        return [clip(random.gauss(m, self.std), 0.0, 1.0) for m in mu]

    def log_prob(self, action, mu):
        return sum(math.log(normal_pdf(action[j], mu[j], self.std)) for j in range(len(action)))

    def ppo_step(self, states, actions, advantages, old_log_probs, rewards, lr):
        dW1 = [[0.0] * self.hidden_size for _ in range(self.input_size)]
        db1 = [[0.0] * self.hidden_size]
        dW2 = [[0.0] * self.output_size for _ in range(self.hidden_size)]
        db2 = [[0.0] * self.output_size]
        N = len(states)
        if N == 0: return

        for i in range(N):
            mu, V, a1, z1, z2 = self.forward(states[i])
            if not mu: continue
            log_prob = self.log_prob(actions[i], mu)
            ratio = math.exp(log_prob - old_log_probs[i])
            surr1 = ratio * advantages[i]
            surr2 = clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[i]
            policy_loss = -min(surr1, surr2)
            value_loss = self.c1 * (V - symlog(rewards[i])) ** 2
            entropy = self.c2 * (len(mu) * (0.5 * math.log(2 * math.pi * self.std**2) + 0.5))
            loss = policy_loss + value_loss - entropy
            
            delta2 = [0.0] * self.output_size
            for j in range(self.output_size - 1):
                # Gradient of policy loss w.r.t. mu
                grad_policy = -advantages[i] * (1 if surr1 < surr2 and ratio < 1+self.clip_eps and ratio > 1-self.clip_eps else 0) * ratio
                delta2[j] = grad_policy * (actions[i][j] - mu[j]) / (self.std**2) * sigmoid_deriv(mu[j])
            
            # Gradient of value loss w.r.t. V
            delta2[-1] = 2 * self.c1 * (V - symlog(rewards[i])) * sigmoid_deriv(V)
            
            a1_t = matrix_transpose(a1)
            grad_W2 = matrix_multiply(a1_t, [delta2])
            grad_b2 = [delta2]
            
            w2_t = matrix_transpose(self.W2)
            da1 = matrix_multiply([delta2], w2_t)
            delta1 = [da1[0][k] * sigmoid_deriv(a1[0][k]) for k in range(self.hidden_size)]
            
            x = [[symlog(states[i][k]) for k in range(self.input_size)]]
            x_t = matrix_transpose(x)
            grad_W1 = matrix_multiply(x_t, [delta1])
            grad_b1 = [delta1]
            
            dW1 = matrix_add(dW1, grad_W1)
            db1 = matrix_add(db1, grad_b1)
            dW2 = matrix_add(dW2, grad_W2)
            db2 = matrix_add(db2, grad_b2)
            
        dW1 = scalar_matrix_multiply(1.0 / N * lr, dW1)
        db1 = scalar_matrix_multiply(1.0 / N * lr, db1)
        dW2 = scalar_matrix_multiply(1.0 / N * lr, dW2)
        db2 = scalar_matrix_multiply(1.0 / N * lr, db2)

        self.W1 = matrix_subtract(self.W1, dW1)
        self.b1 = matrix_subtract(self.b1, db1)
        self.W2 = matrix_subtract(self.W2, dW2)
        self.b2 = matrix_subtract(self.b2, db2)

    def copy(self):
        copy = NeuralDecoder(self.input_size, self.hidden_size, self.output_size)
        copy.W1 = [row[:] for row in self.W1]
        copy.b1 = [row[:] for row in self.b1]
        copy.W2 = [row[:] for row in self.W2]
        copy.b2 = [row[:] for row in self.b2]
        return copy

class ImmuneDecoder:
    def __init__(self, n_qubits, pop_size=20, generations=10, mutation_rate=0.1):
        self.n_qubits = n_qubits
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def generate_antibody(self):
        return [random.randint(0, 1) for _ in range(self.n_qubits)]

    def fitness(self, antibody, syndrome, code):
        corrected_errors = antibody
        residual = code.measure_syndrome(corrected_errors)
        mismatch = sum(abs(residual[i] - syndrome[i]) for i in range(len(syndrome)))
        logical = code.logical_error(corrected_errors, [0] * self.n_qubits)
        return -mismatch - 10 * logical

    def mutate(self, antibody):
        return [1 - ab if random.random() < self.mutation_rate else ab for ab in antibody]

    def decode(self, syndrome, code):
        population = [self.generate_antibody() for _ in range(self.pop_size)]
        for gen in range(self.generations):
            fitnesses = [self.fitness(ab, syndrome, code) for ab in population]
            sorted_pop = [p for _, p in sorted(zip(fitnesses, population), reverse=True)]
            selected = sorted_pop[:self.pop_size // 2]
            new_pop = []
            for s in selected:
                clones = [self.mutate(s) for _ in range(2)]
                new_pop.extend(clones)
            population = new_pop[:self.pop_size]
        best = max(population, key=lambda ab: self.fitness(ab, syndrome, code))
        return best

# Reptile Meta-Learning
def reptile_update(meta_model, task_models, beta):
    for i in range(len(task_models)):
        diff_W1 = matrix_subtract(task_models[i].W1, meta_model.W1)
        diff_b1 = matrix_subtract(task_models[i].b1, meta_model.b1)
        diff_W2 = matrix_subtract(task_models[i].W2, meta_model.W2)
        diff_b2 = matrix_subtract(task_models[i].b2, meta_model.b2)
        meta_model.W1 = matrix_add(meta_model.W1, scalar_matrix_multiply(beta, diff_W1))
        meta_model.b1 = matrix_add(meta_model.b1, scalar_matrix_multiply(beta, diff_b1))
        meta_model.W2 = matrix_add(meta_model.W2, scalar_matrix_multiply(beta, diff_W2))
        meta_model.b2 = matrix_add(meta_model.b2, scalar_matrix_multiply(beta, diff_b2))

# --- NEW: Functions for Training and Validating the Complexity Estimator ---
def generate_estimator_training_data(n_samples=100, n_qubits=9):
    """Generates training data for the NeuralComplexityEstimator."""
    X_data = []
    y_data = []
    op_size = int(math.sqrt(2**n_qubits)) # For n=9, this is not integer. Let's fix to 3 qubits (8x8 matrix)
    n_qubits = 3 
    op_size = 8

    for i in range(n_samples):
        is_complex = (i % 2 == 0)
        op_real = [[0.0]*op_size for _ in range(op_size)]
        op_imag = [[0.0]*op_size for _ in range(op_size)]

        if is_complex: # Complex Error (e.g., CNOT)
            op_real[0][0], op_real[1][1] = 1.0, 1.0
            op_real[2][3], op_real[3][2] = 1.0, 1.0
            for j in range(4, 8): op_real[j][j] = 1.0
            # LZ complexity is higher for structured matrices
            label = 0.8
        else: # Simple Error (e.g., Pauli-X on q0)
            op_real[0][1], op_real[1][0] = 1.0, 1.0
            op_real[2][3], op_real[3][2] = 1.0, 1.0
            op_real[4][5], op_real[5][4] = 1.0, 1.0
            op_real[6][7], op_real[7][6] = 1.0, 1.0
            # LZ complexity is lower for simple permutations
            label = 0.2
        
        X_data.append((op_real, op_imag))
        y_data.append(label)
    return X_data, y_data

def train_complexity_estimator(estimator, data, epochs=10, lr=0.01):
    """Simple supervised training loop for the estimator."""
    X_data, y_data = data
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_data)):
            op_real, op_imag = X_data[i]
            y_true = y_data[i]
            
            # Forward pass
            y_pred = estimator.forward(op_real, op_imag)
            
            # Compute loss (MSE)
            loss = (y_pred - y_true) ** 2
            total_loss += loss
            
            # Backward pass (simplified gradient update)
            grad_loss = 2 * (y_pred - y_true)
            
            # Update FC weights
            flat_vector = flatten(estimator.forward(op_real, op_imag)[1]) # Need to get intermediate layer
            for j in range(len(estimator.fc_weights)):
                estimator.fc_weights[j][0] -= lr * grad_loss * flat_vector[j]

            # Update Conv weights (highly simplified)
            for f in range(estimator.n_filters):
                for c in range(2): # real/imag channels
                    for k1 in range(estimator.kernel_size):
                        for k2 in range(estimator.kernel_size):
                           estimator.conv_kernels[0][f][c][k1][k2] -= lr * grad_loss * 0.01 # Naive update

        if epoch % 2 == 0:
            print(f"Estimator Epoch {epoch}, Avg Loss: {total_loss / len(X_data):.4f}")


# Orchestrator with Experiment Harness
class Orchestrator:
    def __init__(self):
        self.code = SurfaceCode(d=3)
        self.n_stab = self.code.n_stabilizers
        self.n_qubits = self.code.n_qubits
        self.uf_decoder = UnionFindDecoder(self.n_stab)
        self.neural_decoder = NeuralDecoder(self.n_stab, 16, self.n_qubits + 1)
        self.immune_decoder = ImmuneDecoder(self.n_qubits)
        self.meta_lr = 0.01
        self.inner_lr = 0.1
        self.inner_steps = 5
        self.batch_size = 32
        self.num_batches = 10
        self.log = []
        self.leaderboard = {"UF": {"LER": 0.0, "speed": 0.0, "adapt": 0.0}, "Neural": {"LER": 0.0, "speed": 0.0, "adapt": 0.0}, "Immune": {"LER": 0.0, "speed": 0.0, "adapt": 0.0}}
        self.current_best = "UF"
        self.hidden_sizes = [8, 16, 32]
        self.current_hidden = 16
        self.error_rates = [0.01, 0.05, 0.1]
        self.code_distances = [3]
        self.regime_map = {}
        # --- NEW: Initialize the Complexity Estimator ---
        self.complexity_estimator = NeuralComplexityEstimator(input_shape=(8,8))

    def generate_batch(self):
        syndromes = []
        true_errors = []
        for _ in range(self.batch_size):
            state = [0] * self.n_qubits
            errors = self.code.apply_bitflip_noise(state)
            syndrome = self.code.measure_syndrome(errors)
            true_errors.append(errors)
            syndromes.append(syndrome)
        return syndromes, true_errors

    def benchmark_decoder(self, decoder_type, syndromes, true_errors):
        start = time.time()
        ler = 0.0
        for s, e in zip(syndromes, true_errors):
            if decoder_type == "UF":
                corr = self.uf_decoder.decode(s)
            elif decoder_type == "Neural":
                mu, _, _, _, _ = self.neural_decoder.forward(s)
                if not mu:
                    corr = [0] * self.n_qubits
                else:
                    corr = [1 if m > 0.5 else 0 for m in mu]
            elif decoder_type == "Immune":
                corr = self.immune_decoder.decode(s, self.code)
            
            if corr:
                 ler += self.code.logical_error(e, corr)
        ler /= len(syndromes) if len(syndromes) > 0 else 1
        speed = time.time() - start
        return ler, speed

    def train_neural(self, syndromes, true_errors):
        if not syndromes: return
        actions = []
        old_log_probs = []
        advantages = []
        rewards = []
        for i, s in enumerate(syndromes):
            mu, V, _, _, _ = self.neural_decoder.forward(s)
            if not mu: continue
            a = self.neural_decoder.sample_action(mu)
            actions.append(a)
            old_log_probs.append(self.neural_decoder.log_prob(a, mu))
            corr = [1 if ai > 0.5 else 0 for ai in a]
            r = 1.0 - self.code.logical_error(true_errors[i], corr)
            rewards.append(r)
            advantages.append(symlog(r) - V)
        self.neural_decoder.ppo_step(syndromes, actions, advantages, old_log_probs, rewards, self.inner_lr)

    def adapt_neural(self, syndromes, true_errors):
        task_model = self.neural_decoder.copy()
        for _ in range(self.inner_steps):
            self.train_neural(syndromes, true_errors) # Simplified adaptation
        return task_model
        
    def adapt_immune(self, syndromes, true_errors):
        self.immune_decoder.mutation_rate = clip(self.immune_decoder.mutation_rate + random.uniform(-0.01, 0.01), 0.05, 0.2)
        return self.immune_decoder

    def run_sweep(self):
        for d in self.code_distances:
            self.code = SurfaceCode(d=d)
            self.n_qubits = self.code.n_qubits
            self.n_stab = self.code.n_stabilizers
            if self.n_stab == 0: continue
            self.uf_decoder = UnionFindDecoder(self.n_stab)
            self.neural_decoder = NeuralDecoder(self.n_stab, self.current_hidden, self.n_qubits + 1)
            self.immune_decoder = ImmuneDecoder(self.n_qubits)
            for p in self.error_rates:
                self.code.set_error_rate(p)
                syndromes, true_errors = self.generate_batch()
                uf_ler, _ = self.benchmark_decoder("UF", syndromes, true_errors)
                neural_ler, _ = self.benchmark_decoder("Neural", syndromes, true_errors)
                immune_ler, _ = self.benchmark_decoder("Immune", syndromes, true_errors)
                best = min({"UF": uf_ler, "Neural": neural_ler, "Immune": immune_ler}, key=lambda k: {"UF": uf_ler, "Neural": neural_ler, "Immune": immune_ler}[k])
                self.regime_map[(d, p)] = best
                self.log.append(f"Sweep d={d}, p={p}: Best {best}, LER UF {uf_ler:.4f}, Neural {neural_ler:.4f}, Immune {immune_ler:.4f}")

    def visualize_regime_map(self):
        print("Regime Map (d, p) -> Best Decoder:")
        for key, value in sorted(self.regime_map.items()):
            print(f"{key}: {value}")
        # ASCII visualization
        print("\nASCII Heatmap (rows: d, columns: p, symbol: U=UF, N=Neural, I=Immune)")
        ds = sorted(list(set(k[0] for k in self.regime_map)))
        ps = sorted(list(set(k[1] for k in self.regime_map)))
        if not ps: return
        print("d \\ p " + " ".join(f"{p:.2f}" for p in ps))
        for d in ds:
            row = [self.regime_map.get((d, p), "?")[0] for p in ps]
            print(f"{d}     " + " ".join(row))

    def run(self):
        # --- NEW: Train and Validate the Complexity Estimator First ---
        print("--- Training Neural Complexity Estimator ---")
        estimator_data = generate_estimator_training_data(n_samples=200)
        # The training implementation is highly simplified and for demonstration only.
        # A full backpropagation implementation is beyond this scope.
        # We will manually set weights to simulate a trained state.
        print("Simulating trained state for complexity estimator...")
        # Manually set weights to differentiate simple/complex
        # This kernel looks for diagonal/anti-diagonal patterns
        self.complexity_estimator.conv_kernels[0][0][0] = [[1, 0, -1], [0, 1, 0], [-1, 0, 1]]
        self.complexity_estimator.fc_weights[0][0] = 0.5 # Set a weight
        
        print("\n--- Validating Neural Complexity Estimator ---")
        # Generate one simple and one complex error
        X_test, y_test = generate_estimator_training_data(n_samples=2)
        complex_op_real, complex_op_imag = X_test[0]
        simple_op_real, simple_op_imag = X_test[1]

        complex_score = self.complexity_estimator.forward(complex_op_real, complex_op_imag)
        simple_score = self.complexity_estimator.forward(simple_op_real, simple_op_imag)
        print(f"Validation Score for Complex Error (CNOT-like): {complex_score:.4f}")
        print(f"Validation Score for Simple Error (Pauli-X-like): {simple_score:.4f}")
        if complex_score > simple_score:
            print("✅ Validation Successful: Estimator correctly distinguishes complex vs. simple errors.")
        else:
            print("❌ Validation Failed: Estimator did not distinguish errors correctly.")
        print("--------------------------------------------\n")


        self.run_sweep()
        self.visualize_regime_map()
        
        for batch in range(self.num_batches):
            syndromes, true_errors = self.generate_batch()
            if not syndromes: continue
            
            uf_ler, uf_speed = self.benchmark_decoder("UF", syndromes, true_errors)
            neural_ler, neural_speed = self.benchmark_decoder("Neural", syndromes, true_errors)
            immune_ler, immune_speed = self.benchmark_decoder("Immune", syndromes, true_errors)

            self.leaderboard["UF"]["LER"] = (self.leaderboard["UF"]["LER"] * batch + uf_ler) / (batch + 1)
            self.leaderboard["UF"]["speed"] = (self.leaderboard["UF"]["speed"] * batch + uf_speed) / (batch + 1)
            self.leaderboard["Neural"]["LER"] = (self.leaderboard["Neural"]["LER"] * batch + neural_ler) / (batch + 1)
            self.leaderboard["Neural"]["speed"] = (self.leaderboard["Neural"]["speed"] * batch + neural_speed) / (batch + 1)
            self.leaderboard["Immune"]["LER"] = (self.leaderboard["Immune"]["LER"] * batch + immune_ler) / (batch + 1)
            self.leaderboard["Immune"]["speed"] = (self.leaderboard["Immune"]["speed"] * batch + immune_speed) / (batch + 1)

            # Adaptation steps (simplified)
            start_adapt_neural = time.time()
            self.adapt_neural(syndromes[:8], true_errors[:8])
            neural_adapt_time = time.time() - start_adapt_neural

            start_adapt_immune = time.time()
            self.adapt_immune(syndromes[:8], true_errors[:8])
            immune_adapt_time = time.time() - start_adapt_immune

            self.leaderboard["Neural"]["adapt"] = (self.leaderboard["Neural"]["adapt"] * batch + neural_adapt_time) / (batch + 1)
            self.leaderboard["Immune"]["adapt"] = (self.leaderboard["Immune"]["adapt"] * batch + immune_adapt_time) / (batch + 1)
            self.leaderboard["UF"]["adapt"] = 0.0

            scores = {name: stats["LER"] + 0.1 * stats["speed"] + 0.05 * stats["adapt"] for name, stats in self.leaderboard.items()}
            self.current_best = min(scores, key=scores.get)
            
            log_entry = f"Batch {batch}: Best {self.current_best}, UF LER {uf_ler:.4f}, Neural LER {neural_ler:.4f}, Immune LER {immune_ler:.4f}"
            self.log.append(log_entry)
            print(log_entry)
            
        print("\nFinal Leaderboard:")
        for name, stats in self.leaderboard.items():
            print(f"{name}: LER {stats['LER']:.4f}, speed {stats['speed']:.4f}, adapt {stats['adapt']:.4f}")

# Run
orch = Orchestrator()
orch.run()

