"""
Exercise 10: Filter Explorer Lab

This script demonstrates how convolution filters work by applying handcrafted
filters to images and visualizing the results.

HOW TO RUN:
1. Install required dependencies:
   pip install -r requirements.txt
   
   Or install individually:
   pip install numpy matplotlib tensorflow

2. Run the script:
   python3 ex10-main.py

NOTE: Multiple matplotlib windows will open. Close each window to proceed
      to the next visualization, or run in a non-interactive environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ================================================
# Part 1: Load and Display the Image
# ================================================

# Load a sample image (use a built-in or your own)
# Option A: Use TensorFlow's sample image
sample = tf.keras.preprocessing.image.load_img(
    tf.keras.utils.get_file('cat.jpg', 
    'https://i.guim.co.uk/img/media/327aa3f0c3b8e40ab03b4ae80319064e401c6fbc/377_133_3542_2834/master/3542.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=34d32522f47e4a67286f9894fc81c863'),
    target_size=(224, 224)
)
image = np.array(sample)

# Convert to grayscale for simpler filtering
gray = np.mean(image, axis=2)

# TODO: Display the original and grayscale images
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
plt.show()

# ================================================
# Part 2: Define Classic Filters
# ================================================
# Task 2.1: Edge Detection Filters
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# TODO: Create a simple vertical edge detector
vertical_edge = np.array([
    # Your 3x3 filter
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# TODO: Create a simple horizontal edge detector
horizontal_edge = np.array([
    # Your 3x3 filter
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# ================================================
# Task 2.2: Other Filters

# Sharpening filter
sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Blur filter (box blur)
blur = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

# TODO: Create a Gaussian blur (3x3 approximation)
gaussian_blur = np.array([
    # Hint: Center should be highest, corners lowest
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
], dtype=np.float32)

# Emboss filter
emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.float32)

# ================================================
# Part 3: Apply Filters to the Image
# ================================================

# Task 3.1: Create Convolution Function
def apply_filter(image, kernel):
    """
    Apply a single filter to a grayscale image using TensorFlow.
    
    Args:
        image: 2D numpy array (height, width)
        kernel: 2D numpy array (kh, kw)
    
    Returns:
        Filtered image
    """
    # Reshape for TensorFlow: (batch, height, width, channels)
    img_tensor = image.reshape(1, image.shape[0], image.shape[1], 1)
    img_tensor = tf.cast(img_tensor, tf.float32)
    
    # Reshape kernel: (kh, kw, in_channels, out_channels)
    kernel_tensor = kernel.reshape(kernel.shape[0], kernel.shape[1], 1, 1)
    kernel_tensor = tf.cast(kernel_tensor, tf.float32)
    
    # Apply convolution
    output = tf.nn.conv2d(img_tensor, kernel_tensor, strides=1, padding='SAME')
    
    return output.numpy().squeeze()

# TODO: Test with sobel_x
filtered = apply_filter(gray, sobel_x)
plt.imshow(filtered, cmap='gray')
plt.title('Sobel X (Vertical Edges)')
plt.show()

# Task 3.2: Apply All Filters
filters = {
    'Sobel X': sobel_x,
    'Sobel Y': sobel_y,
    'Sharpen': sharpen,
    'Blur': blur,
    'Emboss': emboss
}

# TODO: Create a grid showing all filter results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original')

for i, (name, kernel) in enumerate(filters.items(), start=1):
    filtered = apply_filter(gray, kernel)
    axes[i].imshow(filtered, cmap='gray')
    axes[i].set_title(name)

plt.tight_layout()
plt.show()

# ================================================
# Part 4: Edge Magnitude
# ================================================

# Apply Sobel filters to get edge responses in X and Y directions
edges_x = apply_filter(gray, sobel_x)
edges_y = apply_filter(gray, sobel_y)

# Calculate edge magnitude: sqrt(edges_x^2 + edges_y^2)
edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)

# Display original vs edge magnitude
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(edge_magnitude, cmap='gray')
axes[1].set_title('Edge Magnitude')
plt.show()


# ================================================
# Part 5: Create Your Own Filters (Bonus)
# ================================================

# Challenge A: Diagonal Edge Detector (/)
# Detects diagonal edges going from top-left to bottom-right
diagonal_filter = np.array([
    [-2, -1,  0],
    [-1,  0,  1],
    [ 0,  1,  2]
], dtype=np.float32)

# Alternative: Detects diagonal edges going from top-right to bottom-left (\)
diagonal_filter_alt = np.array([
    [ 0, -1, -2],
    [ 1,  0, -1],
    [ 2,  1,  0]
], dtype=np.float32)

# Challenge B: Corner Detector
# Detects corners by looking for patterns where edges meet
# This filter responds strongly to top-left corners
corner_filter = np.array([
    [-1, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  1]
], dtype=np.float32)

# Alternative corner detectors for different corner orientations
corner_top_right = np.array([
    [ 0, -1, -1],
    [ 1,  1, -1],
    [ 1,  1,  0]
], dtype=np.float32)

corner_bottom_left = np.array([
    [ 0,  1,  1],
    [-1,  1,  1],
    [-1, -1,  0]
], dtype=np.float32)

corner_bottom_right = np.array([
    [ 1,  1,  0],
    [ 1,  1, -1],
    [ 0, -1, -1]
], dtype=np.float32)

# Challenge C: Custom Pattern - Horizontal Line Detector
# Detects horizontal lines (useful for detecting text lines, horizons, etc.)
horizontal_line = np.array([
    [-1, -1, -1],
    [ 2,  2,  2],
    [-1, -1, -1]
], dtype=np.float32)

# Custom Pattern - Vertical Line Detector
vertical_line = np.array([
    [-1,  2, -1],
    [-1,  2, -1],
    [-1,  2, -1]
], dtype=np.float32)

# Custom Pattern - Spot Detector (detects bright spots/dots)
spot_detector = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Test the custom filters
custom_filters = {
    'Diagonal (/)': diagonal_filter,
    'Corner (top-left)': corner_filter,
    'Horizontal Line': horizontal_line,
    'Vertical Line': vertical_line,
    'Spot Detector': spot_detector
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original')

for i, (name, kernel) in enumerate(custom_filters.items(), start=1):
    filtered = apply_filter(gray, kernel)
    axes[i].imshow(filtered, cmap='gray')
    axes[i].set_title(name)

plt.tight_layout()
plt.show()

# ================================================
# Part 6: Reflection
# ================================================

print("=" * 60)
print("REFLECTION QUESTIONS")
print("=" * 60)

# Q1: What do positive and negative values in a filter represent?
print("\nQ1: What do positive and negative values in a filter represent?")
print("Answer:")
print("Positive values in a filter represent areas where pixel values should be")
print("increased (brightened), while negative values represent areas where pixel")
print("values should be decreased (darkened). When working with edge detection filters, positive")
print("and negative values create contrast by comparing different regions of the")
print("image. When a filter is applied, areas with matching patterns produce")
print("strong responses (high absolute values), while uniform areas produce weak")
print("responses (near zero).")

# Q2: Why does the blur filter have all positive values that sum to 1?
print("\nQ2: Why does the blur filter have all positive values that sum to 1?")
print("Answer:")
print("The blur filter has all positive values that sum to 1 to preserve the")
print("overall brightness/intensity of the image. When you average pixel values")
print("(which is what blur does), the sum of weights must equal 1 to maintain")
print("the same average intensity. If the sum were greater than 1, the image")
print("would get brighter; if less than 1, it would get darker. By summing to")
print("1, blur only smooths the image without changing its overall brightness level.")

# Q3: Looking at the Sobel results, why does X detect vertical edges
#     and Y detect horizontal edges? (It seems backwards!)
print("\nQ3: Looking at the Sobel results, why does X detect vertical edges")
print("and Y detect horizontal edges? (It seems backwards!)")
print("Answer:")
print("Sobel X detects vertical edges because it measures the gradient (rate of")
print("change) in the X (horizontal) direction. A horizontal edge has a strong vertical gradient")
print("(pixels change rapidly from top to bottom). The naming refers to the")
print("direction of the gradient being measured, not the orientation of the edge.")

# Q4: How do you think CNN filters differ from these handcrafted ones?
print("\nQ4: How do you think CNN filters differ from these handcrafted ones?")
print("Answer:")
print("1. Learning: CNN filters are learned automatically from data through")
print("   backpropagation, rather than being manually designed by humans.")
print("2. Complexity: CNNs can learn complex, non-intuitive patterns that might")
print("   not be obvious to human designers.")

print("\n" + "=" * 60)