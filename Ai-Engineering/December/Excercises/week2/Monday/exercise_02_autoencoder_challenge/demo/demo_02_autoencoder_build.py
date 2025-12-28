"""
Demo 02: Building and Training an Autoencoder

This demo shows trainees how to:
1. Build a simple autoencoder architecture
2. Train the autoencoder for image compression/reconstruction
3. Visualize the encoding (latent space) and decoding (reconstruction)
4. Understand the compression-reconstruction tradeoff

Learning Objectives:
- Understand encoder-decoder architecture
- Learn how autoencoders compress information
- Visualize latent space representations
- Apply autoencoders to image reconstruction

References:
- Written Content: 02-autoencoders-concepts.md
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# ============================================================================
# PART 1: Load and Prepare MNIST Data
# ============================================================================

print("=" * 70)
print("PART 1: Loading MNIST Dataset for Autoencoder Training")
print("=" * 70)

# Load MNIST
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalize to [0, 1] range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images for autoencoder input
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

print(f"\nDataset prepared:")
print(f"  Training samples: {x_train_flat.shape[0]}")
print(f"  Input dimension: {x_train_flat.shape[1]} (28x28 flattened)")
print(f"  Pixel value range: [{x_train_flat.min():.1f}, {x_train_flat.max():.1f}]")

# ============================================================================
# PART 2: Build Autoencoder Architecture
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Building Autoencoder Architecture")
print("=" * 70)

# Define dimensions
input_dim = 784      # 28x28 pixels
encoding_dim = 32    # Compressed representation (24x compression!)

print(f"\nArchitecture Design:")
print(f"  Input dimension: {input_dim}")
print(f"  Encoding dimension: {encoding_dim}")
print(f"  Compression ratio: {input_dim / encoding_dim:.1f}x")

# Build Encoder
print("\n[Encoder] Building compression pathway...")
encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
encoded = layers.Dense(128, activation='relu', name='encoder_layer1')(encoder_input)
encoded = layers.Dense(64, activation='relu', name='encoder_layer2')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu', name='latent_space')(encoded)

encoder = keras.Model(encoder_input, encoded, name='encoder')
print("[OK] Encoder created: 784 -> 128 -> 64 -> 32")

# Build Decoder
print("\n[Decoder] Building reconstruction pathway...")
decoder_input = layers.Input(shape=(encoding_dim,), name='decoder_input')
decoded = layers.Dense(64, activation='relu', name='decoder_layer1')(decoder_input)
decoded = layers.Dense(128, activation='relu', name='decoder_layer2')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid', name='reconstruction')(decoded)

decoder = keras.Model(decoder_input, decoded, name='decoder')
print("[OK] Decoder created: 32 -> 64 -> 128 -> 784")

# Build Complete Autoencoder
print("\n[Autoencoder] Connecting encoder and decoder...")
autoencoder_input = layers.Input(shape=(input_dim,), name='autoencoder_input')
encoded_repr = encoder(autoencoder_input)
reconstructed = decoder(encoded_repr)

autoencoder = keras.Model(autoencoder_input, reconstructed, name='autoencoder')

print("\nComplete Autoencoder Summary:")
autoencoder.summary()

# ============================================================================
# PART 3: Compile and Train Autoencoder
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Training the Autoencoder")
print("=" * 70)

# Compile model
print("\n[Step 1] Compiling model...")
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Good for normalized [0,1] data
    metrics=['mse']              # Track mean squared error
)

print("[OK] Optimizer: Adam")
print("[OK] Loss function: Binary Cross-Entropy")
print("  (Measures reconstruction quality)")

# Train autoencoder
print("\n[Step 2] Training autoencoder...")
print("INSTRUCTOR NOTE: Input = Output (unsupervised learning!)")

# Set up TensorBoard logging
log_dir = "logs/autoencoder/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

print(f"TensorBoard logs: {log_dir}")
print("View with: tensorboard --logdir=logs/autoencoder")

history = autoencoder.fit(
    x_train_flat, x_train_flat,  # Input = Output!
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat),
    callbacks=[tensorboard_callback],
    verbose=1
)

# ============================================================================
# PART 4: Training Progress (View in TensorBoard)
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Analyzing Training Progress")
print("=" * 70)

print("\nTraining metrics are now logged to TensorBoard!")
print(f"To view training curves, run:")
print(f"  tensorboard --logdir=logs/autoencoder")
print(f"  Then navigate to http://localhost:6006")
print("\nAvailable in TensorBoard:")
print("  - SCALARS: loss, val_loss, mse, val_mse")
print("  - HISTOGRAMS: Weight distributions per epoch")
print("  - GRAPHS: Model architecture")

# ============================================================================
# PART 5: Visualize Reconstructions
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Visualizing Image Reconstructions")
print("=" * 70)

# Select test images
n_images = 10
test_images = x_test_flat[:n_images]

# Encode and decode
print("\n[Step 1] Encoding images to 32-dimensional latent space...")
encoded_imgs = encoder.predict(test_images, verbose=0)
print(f"Latent representations shape: {encoded_imgs.shape}")
print(f"Example latent code (first 10 values): {encoded_imgs[0][:10]}")

print("\n[Step 2] Decoding latent representations back to images...")
reconstructed_imgs = autoencoder.predict(test_images, verbose=0)

# Visualize original vs reconstructed
fig, axes = plt.subplots(2, n_images, figsize=(20, 4))

for i in range(n_images):
    # Original images
    axes[0, i].imshow(test_images[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=12, fontweight='bold')
    
    # Reconstructed images
    axes[1, i].imshow(reconstructed_imgs[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Reconstructed', fontsize=12, fontweight='bold')

plt.suptitle('Autoencoder: Original vs Reconstructed Images (32x Compression)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('autoencoder_reconstruction.png', dpi=150)
print("\n[OK] Reconstruction visualization saved to: autoencoder_reconstruction.png")

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(test_images - reconstructed_imgs), axis=1)
print(f"\nAverage reconstruction error (MSE): {np.mean(reconstruction_error):.6f}")
print(f"Min reconstruction error: {np.min(reconstruction_error):.6f}")
print(f"Max reconstruction error: {np.max(reconstruction_error):.6f}")

# ============================================================================
# PART 6: Explore Latent Space
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Exploring the 32-Dimensional Latent Space")
print("=" * 70)

# Encode full test set
print("\n[Step 1] Encoding all test images...")
all_encoded = encoder.predict(x_test_flat, verbose=0)

# Visualize latent space using t-SNE (reduce 32D -> 2D)
from sklearn.manifold import TSNE

print("\n[Step 2] Reducing 32D latent space to 2D using t-SNE...")
print("INSTRUCTOR NOTE: This takes ~30 seconds, good time to explain t-SNE")

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_2d = tsne.fit_transform(all_encoded[:1000])  # Use first 1000 for speed

# Get labels for coloring
test_labels = keras.datasets.mnist.load_data()[1][1][:1000]

# Plot latent space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=test_labels, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter, label='Digit Class')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Autoencoder Latent Space Visualization (32D -> 2D)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('latent_space_tsne.png', dpi=150)
print("[OK] Latent space visualization saved to: latent_space_tsne.png")

print("\nObservations:")
print("- Similar digits cluster together in latent space")
print("- Autoencoder learned meaningful representations")
print("- 32 dimensions capture essential digit features")

# ============================================================================
# PART 7: Denoising Autoencoder (Bonus)
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Denoising Autoencoder (Bonus Demo)")
print("=" * 70)

# Add noise to images
noise_factor = 0.3
x_train_noisy = x_train_flat + noise_factor * np.random.normal(size=x_train_flat.shape)
x_test_noisy = x_test_flat + noise_factor * np.random.normal(size=x_test_flat.shape)

# Clip values to [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

print("\n[Step 1] Added noise to images...")
print(f"Noise factor: {noise_factor}")

# Train denoising autoencoder
print("\n[Step 2] Training denoising autoencoder...")
print("INSTRUCTOR NOTE: Input = noisy, Output = clean (supervised denoising)")

# Build fresh autoencoder for denoising
denoising_autoencoder = keras.Model(autoencoder_input, reconstructed, name='denoising_autoencoder')
denoising_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# TensorBoard for denoising
log_dir_denoising = "logs/autoencoder_denoising/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir_denoising, exist_ok=True)
tb_denoising = keras.callbacks.TensorBoard(log_dir=log_dir_denoising, histogram_freq=1)

history_denoising = denoising_autoencoder.fit(
    x_train_noisy, x_train_flat,  # Input = noisy, Output = clean
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test_flat),
    callbacks=[tb_denoising],
    verbose=0
)

print("[OK] Denoising autoencoder trained")
print(f"Denoising logs: {log_dir_denoising}")

# Visualize denoising results
n_images = 10
test_noisy = x_test_noisy[:n_images]
test_clean = x_test_flat[:n_images]
denoised = denoising_autoencoder.predict(test_noisy, verbose=0)

fig, axes = plt.subplots(3, n_images, figsize=(20, 6))

for i in range(n_images):
    # Noisy images
    axes[0, i].imshow(test_noisy[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Noisy', fontsize=12, fontweight='bold')
    
    # Denoised images
    axes[1, i].imshow(denoised[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Denoised', fontsize=12, fontweight='bold')
    
    # Original clean images
    axes[2, i].imshow(test_clean[i].reshape(28, 28), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Original', fontsize=12, fontweight='bold')

plt.suptitle('Denoising Autoencoder: Removing Noise from Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('denoising_autoencoder.png', dpi=150)
print("[OK] Denoising visualization saved to: denoising_autoencoder.png")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Autoencoder Build")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Autoencoders compress data through a bottleneck")
print("2. Encoder learns compressed representation (784 -> 32)")
print("3. Decoder reconstructs from compressed representation (32 -> 784)")
print("4. Training objective: minimize reconstruction error")
print("5. Latent space captures meaningful features")
print("6. Denoising autoencoders remove noise by training on corrupted inputs")

print("\nGenerated Visualizations:")
print("[TB] TensorBoard: Training curves (tensorboard --logdir=logs/autoencoder)")
print("[OK] autoencoder_reconstruction.png - Original vs reconstructed")
print("[OK] latent_space_tsne.png - 2D visualization of latent space")
print("[OK] denoising_autoencoder.png - Noise removal demonstration")

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)
print("\nDiscussion Points:")
print("1. Why does autoencoder use sigmoid activation in output?")
print("   -> Output must be [0,1] like normalized pixels")
print("2. What happens with smaller encoding dimensions (e.g., 16, 8)?")
print("   -> More compression, worse reconstruction quality")
print("3. What happens with larger encoding dimensions (e.g., 64, 128)?")
print("   -> Less compression, better reconstruction (but less useful)")
print("4. Why is reconstruction slightly blurry?")
print("   -> Lossy compression discards fine details")

print("\nInteractive Exercise Ideas:")
print("- Have trainees modify encoding_dim and observe quality changes")
print("- Try different activation functions in bottleneck layer")
print("- Experiment with different noise levels for denoising")
print("- Apply autoencoder to Fashion-MNIST (different domain)")

print("\n" + "=" * 70)