"""
Demo 06: Forward Propagation Trace
==================================
Week 1, Wednesday - Neural Network Fundamentals

Step-by-step forward propagation through a small MLP.
Every intermediate value is printed for learning.

INSTRUCTOR NOTES:
- Have the MLP diagram open side-by-side
- Trace each value through the network on the board
- This prepares trainees for tomorrow's TensorFlow implementation

Estimated Time: 15-20 minutes
"""

import numpy as np

# =============================================================================
# SECTION 1: NETWORK ARCHITECTURE
# =============================================================================
print("=" * 60)
print("DEMO 06: FORWARD PROPAGATION TRACE")
print("Following Data Through a Neural Network")
print("=" * 60)

print("""
NETWORK ARCHITECTURE:

    Input Layer (3 features)
          |
          v
    Hidden Layer 1 (4 neurons, ReLU)
          |
          v
    Hidden Layer 2 (2 neurons, ReLU)
          |
          v
    Output Layer (1 neuron, Sigmoid)

Total: 3 -> 4 -> 2 -> 1
""")

# =============================================================================
# SECTION 2: INITIALIZE NETWORK
# =============================================================================
print("\n--- INITIALIZING NETWORK ---")

np.random.seed(42)

# Layer 1: 3 inputs -> 4 neurons
W1 = np.array([
    [0.2, -0.3, 0.4, 0.1],
    [0.5, 0.1, -0.2, 0.3],
    [-0.1, 0.4, 0.2, -0.3]
])
b1 = np.array([0.1, -0.1, 0.2, 0.0])

# Layer 2: 4 inputs -> 2 neurons
W2 = np.array([
    [0.3, -0.2],
    [0.4, 0.1],
    [-0.3, 0.5],
    [0.2, -0.1]
])
b2 = np.array([0.1, -0.1])

# Layer 3 (Output): 2 inputs -> 1 neuron
W3 = np.array([
    [0.5],
    [-0.3]
])
b3 = np.array([0.1])

print("Weight Matrices:")
print(f"W1 shape: {W1.shape} (3 inputs x 4 neurons)")
print(f"W2 shape: {W2.shape} (4 inputs x 2 neurons)")
print(f"W3 shape: {W3.shape} (2 inputs x 1 output)")

print(f"\nBias Vectors:")
print(f"b1: {b1}")
print(f"b2: {b2}")
print(f"b3: {b3}")

# =============================================================================
# SECTION 3: ACTIVATION FUNCTIONS
# =============================================================================
def relu(z):
    """ReLU activation."""
    return np.maximum(0, z)

def sigmoid(z):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-z))

# =============================================================================
# SECTION 4: FORWARD PASS - STEP BY STEP
# =============================================================================
print("\n" + "=" * 60)
print("FORWARD PROPAGATION - DETAILED TRACE")
print("=" * 60)

# Input
X = np.array([0.5, 0.8, -0.2])

print(f"\n>>> INPUT LAYER <<<")
print(f"X = {X}")
print(f"Shape: {X.shape}")

# -------------------------
# LAYER 1: Input -> Hidden1
# -------------------------
print(f"\n>>> LAYER 1: Input -> Hidden Layer 1 <<<")
print(f"Operation: z1 = X @ W1 + b1")

# Step-by-step calculation
print(f"\nCalculating z1[0]:")
print(f"  = X[0]*W1[0,0] + X[1]*W1[1,0] + X[2]*W1[2,0] + b1[0]")
print(f"  = {X[0]}*{W1[0,0]} + {X[1]}*{W1[1,0]} + {X[2]}*{W1[2,0]} + {b1[0]}")
print(f"  = {X[0]*W1[0,0]:.4f} + {X[1]*W1[1,0]:.4f} + {X[2]*W1[2,0]:.4f} + {b1[0]}")
z1_0 = X[0]*W1[0,0] + X[1]*W1[1,0] + X[2]*W1[2,0] + b1[0]
print(f"  = {z1_0:.4f}")

# Full z1 calculation
z1 = np.dot(X, W1) + b1
print(f"\nFull z1 = {z1}")

# Apply ReLU
a1 = relu(z1)
print(f"\nApplying ReLU activation:")
print(f"a1 = ReLU(z1) = max(0, z1)")
print(f"a1 = {a1}")
print(f"(Note: Negative values become 0)")

# -------------------------
# LAYER 2: Hidden1 -> Hidden2
# -------------------------
print(f"\n>>> LAYER 2: Hidden Layer 1 -> Hidden Layer 2 <<<")
print(f"Operation: z2 = a1 @ W2 + b2")

z2 = np.dot(a1, W2) + b2
print(f"\nz2 = {z2}")

a2 = relu(z2)
print(f"\nApplying ReLU activation:")
print(f"a2 = ReLU(z2) = {a2}")

# -------------------------
# LAYER 3: Hidden2 -> Output
# -------------------------
print(f"\n>>> LAYER 3: Hidden Layer 2 -> Output <<<")
print(f"Operation: z3 = a2 @ W3 + b3")

z3 = np.dot(a2, W3) + b3
print(f"\nz3 = {z3}")

# Output uses Sigmoid
output = sigmoid(z3)
print(f"\nApplying Sigmoid activation (for binary classification):")
print(f"y_hat = sigmoid(z3) = 1 / (1 + exp(-z3))")
print(f"y_hat = {output[0]:.6f}")

# =============================================================================
# SECTION 5: SUMMARY VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("FORWARD PASS SUMMARY")
print("=" * 60)

print(f"""
Layer-by-Layer Values:

INPUT:  X = {X}
        |
        v
LAYER 1: z1 = X @ W1 + b1 = {z1}
         a1 = ReLU(z1)    = {a1}
        |
        v
LAYER 2: z2 = a1 @ W2 + b2 = {z2}
         a2 = ReLU(z2)     = {a2}
        |
        v
OUTPUT:  z3 = a2 @ W3 + b3 = {z3}
         y_hat = sigmoid(z3) = {output[0]:.6f}

PREDICTION: {output[0]:.2%} probability of positive class
""")

# =============================================================================
# SECTION 6: PARAMETER COUNT
# =============================================================================
print("\n--- PARAMETER COUNT ---")

params_W1 = W1.size
params_b1 = b1.size
params_W2 = W2.size
params_b2 = b2.size
params_W3 = W3.size
params_b3 = b3.size

total_params = params_W1 + params_b1 + params_W2 + params_b2 + params_W3 + params_b3

print(f"Layer 1: {params_W1} weights + {params_b1} biases = {params_W1 + params_b1}")
print(f"Layer 2: {params_W2} weights + {params_b2} biases = {params_W2 + params_b2}")
print(f"Layer 3: {params_W3} weights + {params_b3} biases = {params_W3 + params_b3}")
print(f"TOTAL: {total_params} trainable parameters")

# =============================================================================
# SECTION 7: BATCH PROCESSING
# =============================================================================
print("\n--- BATCH PROCESSING ---")
print("Real networks process many samples at once!")

# Batch of 3 samples
X_batch = np.array([
    [0.5, 0.8, -0.2],   # Sample 1
    [0.1, 0.3, 0.5],    # Sample 2
    [-0.3, 0.9, 0.2]    # Sample 3
])

print(f"Batch input shape: {X_batch.shape}")
print(f"Input batch:\n{X_batch}")

# Forward pass on batch
z1_batch = np.dot(X_batch, W1) + b1
a1_batch = relu(z1_batch)

z2_batch = np.dot(a1_batch, W2) + b2
a2_batch = relu(z2_batch)

z3_batch = np.dot(a2_batch, W3) + b3
output_batch = sigmoid(z3_batch)

print(f"\nOutput predictions:")
for i, pred in enumerate(output_batch):
    print(f"  Sample {i+1}: {pred[0]:.4f}")

# =============================================================================
# SECTION 8: CLEAN FORWARD FUNCTION
# =============================================================================
print("\n--- CLEAN FORWARD PASS FUNCTION ---")

def forward_pass(X, weights, biases, verbose=False):
    """
    Complete forward pass through the network.
    
    Args:
        X: Input features (batch_size, n_features)
        weights: List of weight matrices [W1, W2, W3]
        biases: List of bias vectors [b1, b2, b3]
        verbose: Print intermediate values
    
    Returns:
        output: Final predictions
        cache: Intermediate values (for backprop)
    """
    cache = {'A0': X}
    A = X
    n_layers = len(weights)
    
    for i, (W, b) in enumerate(zip(weights, biases)):
        # Linear transformation
        Z = np.dot(A, W) + b
        cache[f'Z{i+1}'] = Z
        
        # Activation
        if i < n_layers - 1:  # Hidden layers: ReLU
            A = relu(Z)
        else:  # Output layer: Sigmoid
            A = sigmoid(Z)
        
        cache[f'A{i+1}'] = A
        
        if verbose:
            print(f"Layer {i+1}: Z shape = {Z.shape}, A shape = {A.shape}")
    
    return A, cache

# Test the function
print("\nTesting clean forward pass:")
weights = [W1, W2, W3]
biases = [b1, b2, b3]

predictions, cache = forward_pass(X_batch, weights, biases, verbose=True)

print(f"\nFinal predictions:\n{predictions}")

# =============================================================================
# SECTION 9: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. FORWARD PROPAGATION FLOW:
   Input -> [Linear + Activation] -> [Linear + Activation] -> Output
   
   At each layer:
   - z = W @ a_prev + b  (linear transformation)
   - a = activation(z)    (non-linear transformation)

2. SHAPES MATTER:
   - W shape: (input_features, output_neurons)
   - b shape: (output_neurons,)
   - For matrix multiply: inner dimensions must match!

3. ACTIVATION PLACEMENT:
   - Hidden layers: ReLU (or similar)
   - Output layer: Depends on task
     - Binary classification: Sigmoid
     - Multi-class: Softmax
     - Regression: Linear (no activation)

4. BATCH PROCESSING:
   - Process multiple samples simultaneously
   - Same weights applied to all samples
   - Output shape: (batch_size, output_neurons)

5. CACHE INTERMEDIATE VALUES:
   - Needed for backpropagation (Week 2)
   - Store Z and A at each layer

TOMORROW: TensorFlow makes this automatic!
Model.fit() handles forward pass, loss, and weight updates.
""")

