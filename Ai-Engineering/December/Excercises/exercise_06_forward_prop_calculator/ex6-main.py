import numpy as np

print("=" * 60)
print("FORWARD PROPAGATION")
print("=" * 60)

W1 = [[0.2, -0.5],    # Shape: (3, 2)
      [0.3,  0.4],
      [-0.1, 0.2]]

b1 = [0.1, -0.1]      # Shape: (2,)

# Convert to numpy arrays
W1 = np.array(W1)
b1 = np.array(b1)

# Input
X = np.array([1.0, 0.5, -0.5])

print(f"\n>>> INPUT LAYER <<<")
print(f"X = {X}")
print(f"Shape: {X.shape}")

print(f"\n>>> STEP 1: Hidden Layer Linear Transform <<<")
print(f"Calculate z1 = X @ W1 + b1")

# Calculate z1[0] step by step
print(f"\nCalculating z1[0]:")
print(f"  = X[0]*W1[0,0] + X[1]*W1[1,0] + X[2]*W1[2,0] + b1[0]")
print(f"  = {X[0]}*{W1[0,0]} + {X[1]}*{W1[1,0]} + {X[2]}*{W1[2,0]} + {b1[0]}")
