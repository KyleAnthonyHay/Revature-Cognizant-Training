import tensorflow as tf
import numpy as np


# ================================================
# Task 1.1: Create Tensors
# ================================================

# 1. A scalar (rank 0) with value 42
scalar = tf.constant(42)

# 2. A vector (rank 1) with values [1, 2, 3, 4, 5]
vector = tf.constant([1, 2, 3, 4, 5])

# 3. A 3x3 matrix (rank 2) of all ones
matrix_ones = tf.ones([3, 3])

# 4. A 2x3x4 tensor (rank 3) of zeros
tensor_3d = tf.zeros([2, 3, 4])

# 5. A tensor from a NumPy array
np_array = np.array([[1, 2], [3, 4], [5, 6]])
from_numpy = tf.constant(np_array)

# 6. A random normal tensor with shape (100, 10), mean=0, stddev=1
random_normal = tf.random.normal([100, 10], mean=0.0, stddev=1.0)

# 7. A tensor with values from 0 to 99
range_tensor = tf.range(100)

# Print shapes and dtypes for each
tensors = {
    "scalar": scalar,
    "vector": vector,
    "matrix_ones": matrix_ones,
    "tensor_3d": tensor_3d,
    "from_numpy": from_numpy,
    "random_normal": random_normal,
    "range_tensor": range_tensor
}

for name, tensor in tensors.items():
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Rank: {len(tensor.shape)}")
    print(f"  Total elements: {tf.size(tensor).numpy()}")


# ================================================
# Part 2: Shape Manipulation
# ================================================

# Task 2.1: Reshape Operations
print("\n" + "="*50)
print("Task 2.1: Reshape Operations")
print("="*50)

original = tf.range(24)
print(f"\nOriginal: {original.shape}")

shape_a = tf.reshape(original, [4, 6])
print(f"shape_a (4, 6): {shape_a.shape}")

shape_b = tf.reshape(original, [2, 3, 4])
print(f"shape_b (2, 3, 4): {shape_b.shape}")

shape_c = tf.reshape(original, [24, 1])
print(f"shape_c (24, 1): {shape_c.shape}")

shape_d = tf.reshape(original, [1, 24])
print(f"shape_d (1, 24): {shape_d.shape}")

shape_e = tf.reshape(original, [2, 2, 2, 3])
print(f"shape_e (2, 2, 2, 3): {shape_e.shape}")

# Task 2.2: Using -1 for Automatic Dimension
print("\n" + "="*50)
print("Task 2.2: Using -1 for Automatic Dimension")
print("="*50)

reshape_auto = tf.reshape(original, [6, -1])
print(f"\nReshape to (6, -1): {reshape_auto.shape}")
print(f"-1 became: {reshape_auto.shape[1]}")

try:
    reshape_invalid = tf.reshape(original, [8, -1])
    print(f"Reshape to (8, -1): {reshape_invalid.shape}")
except Exception as e:
    print(f"Reshape to (8, -1) failed: {e}")
    print("Reason: 24 is not divisible by 8, so -1 cannot be inferred")

# Task 2.3: Expand and Squeeze
print("\n" + "="*50)
print("Task 2.3: Expand and Squeeze")
print("="*50)

vector = tf.constant([1, 2, 3, 4])
print(f"\nOriginal vector: {vector.shape}")

expanded_0 = tf.expand_dims(vector, axis=0)
print(f"expanded_0 (axis=0): {expanded_0.shape}")

expanded_1 = tf.expand_dims(vector, axis=1)
print(f"expanded_1 (axis=1): {expanded_1.shape}")

squeezed = tf.squeeze(expanded_0)
print(f"squeezed: {squeezed.shape}")

# ================================================
# Part 4: Common Operations
# ================================================

# Task 4.1: Math Operations
print("\n" + "="*50)
print("Task 4.1: Math Operations")
print("="*50)

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(f"\nInput tensor x:\n{x.numpy()}")

mean_all = tf.reduce_mean(x)
print(f"\nmean_all (mean of all elements): {mean_all.numpy()}")

mean_rows = tf.reduce_mean(x, axis=1)
print(f"mean_rows (mean of each row, axis=1): {mean_rows.numpy()}")

mean_cols = tf.reduce_mean(x, axis=0)
print(f"mean_cols (mean of each column, axis=0): {mean_cols.numpy()}")

sum_all = tf.reduce_sum(x)
print(f"sum_all (sum of all elements): {sum_all.numpy()}")

max_val = tf.reduce_max(x)
print(f"max_val (maximum value): {max_val.numpy()}")

argmax_rows = tf.argmax(x, axis=1)
print(f"argmax_rows (index of max in each row): {argmax_rows.numpy()}")

# Task 4.2: Matrix Operations
print("\n" + "="*50)
print("Task 4.2: Matrix Operations")
print("="*50)

A = tf.constant([[1., 2.],
                 [3., 4.]])

B = tf.constant([[5., 6.],
                 [7., 8.]])

print(f"\nMatrix A:\n{A.numpy()}")
print(f"\nMatrix B:\n{B.numpy()}")

matmul_result = tf.matmul(A, B)
print(f"\nmatmul_result (matrix multiplication A @ B):\n{matmul_result.numpy()}")

elementwise = A * B
print(f"\nelementwise (element-wise multiplication A * B):\n{elementwise.numpy()}")

transposed = tf.transpose(A)
print(f"\ntransposed (transpose of A):\n{transposed.numpy()}")

# ================================================
# Part 5: Challenge Problems
# Challenge A: Batch Normalization Prep
# ================================================

print("\n" + "="*50)
print("Challenge A: Batch Normalization Prep")
print("="*50)

images = tf.random.normal([2, 4, 4, 3])
print(f"\nOriginal images shape: {images.shape}")
print(f"Sample values (first image, first pixel): {images[0, 0, 0, :].numpy()}")

channel_mean = tf.reduce_mean(images, axis=[0, 1, 2])
print(f"\nMean for each channel (shape: {channel_mean.shape}): {channel_mean.numpy()}")

channel_std = tf.math.reduce_std(images, axis=[0, 1, 2])
print(f"Std for each channel (shape: {channel_std.shape}): {channel_std.numpy()}")

channel_mean_expanded = tf.reshape(channel_mean, [1, 1, 1, 3])
channel_std_expanded = tf.reshape(channel_std, [1, 1, 1, 3])

normalized_images = (images - channel_mean_expanded) / channel_std_expanded

print(f"\nNormalized images shape: {normalized_images.shape}")

normalized_channel_mean = tf.reduce_mean(normalized_images, axis=[0, 1, 2])
normalized_channel_std = tf.math.reduce_std(normalized_images, axis=[0, 1, 2])

print(f"\nAfter normalization:")
print(f"Mean for each channel: {normalized_channel_mean.numpy()}")
print(f"Std for each channel: {normalized_channel_std.numpy()}")
print(f"\nEach channel now has mean ≈ 0 and std ≈ 1")
