import numpy as np

# Given weights and biases
W1 = np.array([[0.2, -0.5],
               [0.3,  0.4],
               [-0.1, 0.2]])
b1 = np.array([0.1, -0.1])

W2 = np.array([[0.6],
               [-0.3]])
b2 = np.array([0.1])

# Input
X = np.array([1.0, 0.5, -0.5])
# X = np.array([1.0, 0.5, -0.5]) # input 2  [0.0, 1.0, 1.0]
# X = np.array([-1.0, 0.0, 0.0]) # input 3  [-1.0, 0.0, 0.0]
# X = np.array([2.0, 2.0, 2.0]) # input 4  [2.0, 2.0, 2.0]

# Activation functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# TODO: Implement forward propagation
# Step 1: z1 = ...
z1 = np.dot(X, W1) + b1
# Step 2: a1 = ...
a1 = relu(z1)
# Step 3: z2 = ...
z2 = np.dot(a1, W2) + b2
# Step 4: y_hat = ...
y_hat = sigmoid(z2)

def forward_pass(X, weights, biases):
    """
    General forward pass through any network.
    
    Args:
        X: Input features (can be single sample or batch)
        weights: List of weight matrices [W1, W2, ...]
        biases: List of bias vectors [b1, b2, ...]
    
    Returns:
        output: Final prediction
        cache: Dictionary of intermediate values for backprop
    """
    cache = {'A0': X}
    A = X
    n_layers = len(weights)
    
    for i, (W, b) in enumerate(zip(weights, biases)):
        # Linear transformation: z = A @ W + b
        Z = np.dot(A, W) + b
        cache[f'Z{i+1}'] = Z
        
        # Apply activation function
        if i < n_layers - 1:  # Hidden layers: ReLU
            A = relu(Z)
        else:  # Output layer: Sigmoid
            A = sigmoid(Z)
        
        cache[f'A{i+1}'] = A
    
    return A, cache


# TODO: Print intermediate values and compare with your hand calculations
print(f"z1 = {z1}")
print(f"a1 = {a1}")
print(f"z2 = {z2}")
print(f"y_hat = {y_hat}")

print("--------------------------------")
print("Part 4: Build Forward Pass Function (Bonus)")
print("--------------------------------")
# Test with your network
weights = [W1, W2]
biases = [b1, b2]

output, cache = forward_pass(X, weights, biases)
print(f"Output: {output}")
print(f"Cache keys: {cache.keys()}")
# Cache contains: A0, Z1, A1, Z2, A2 (final output)