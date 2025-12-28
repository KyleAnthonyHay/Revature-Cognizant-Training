# Exercise 02: Fashion-MNIST Autoencoder

## Learning Objectives

- Build encoder-decoder architecture for image compression
- Experiment with different bottleneck sizes
- Visualize latent space representations
- Understand the compression vs. reconstruction quality trade-off

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_02_autoencoder_build.py`, you saw an autoencoder compress MNIST digits through a 32-dimensional bottleneck. Now you'll build one for Fashion-MNIST, a more challenging dataset with clothing items.

## The Math You Need

### Autoencoder Architecture

```
# Encoder: Compress input to latent representation
Input (784) -> Dense(256) -> Dense(128) -> Dense(latent_dim)

# Decoder: Reconstruct from latent representation  
Input (latent_dim) -> Dense(128) -> Dense(256) -> Dense(784)
```

### Reconstruction Loss

```
# Binary Cross-Entropy for pixel-wise reconstruction
# Formula: -sum(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))

# Or Mean Squared Error
# Formula: MSE = (1/n) * sum((y_pred - y_true)^2)
```

---

## Part 1: Build the Autoencoder (20 min)

### Task 1.1: Load Fashion-MNIST

Navigate to `starter_code/autoencoder_starter.py` and load the data:

```python
# Fashion-MNIST has 10 classes:
# 0: T-shirt, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat
# 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# TODO: Normalize to [0, 1] and flatten to 784 dimensions
# Hint: x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
```

### Task 1.2: Build the Encoder

Create an encoder that compresses 784 dimensions down to a configurable `latent_dim`:

```python
def build_encoder(latent_dim):
    """
    Build encoder network.
    
    Architecture:
    - Input: 784 dimensions
    - Hidden: 256 neurons, ReLU
    - Hidden: 128 neurons, ReLU  
    - Output: latent_dim neurons, ReLU
    """
    # TODO: Implement using Keras Functional API
    # encoder_input = layers.Input(shape=(784,))
    # x = layers.Dense(256, activation='relu')(encoder_input)
    # ...
    pass
```

### Task 1.3: Build the Decoder

Create a decoder that mirrors the encoder structure:

```python
def build_decoder(latent_dim):
    """
    Build decoder network.
    
    Architecture:
    - Input: latent_dim dimensions
    - Hidden: 128 neurons, ReLU
    - Hidden: 256 neurons, ReLU
    - Output: 784 neurons, Sigmoid (for [0,1] pixel values)
    """
    # TODO: Implement - note the Sigmoid output activation!
    pass
```

### Task 1.4: Combine into Autoencoder

```python
def build_autoencoder(latent_dim):
    """Combine encoder and decoder into full autoencoder."""
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    
    # TODO: Connect encoder output to decoder input
    # autoencoder_input = layers.Input(shape=(784,))
    # encoded = encoder(autoencoder_input)
    # decoded = decoder(encoded)
    # autoencoder = keras.Model(autoencoder_input, decoded)
    
    return encoder, decoder, autoencoder
```

---

## Part 2: Train and Evaluate (20 min)

### Task 2.1: Train with latent_dim=32

```python
latent_dim = 32
encoder, decoder, autoencoder = build_autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# TODO: Train the autoencoder
# Remember: input = output for autoencoders!
# history = autoencoder.fit(x_train, x_train, ...)
```

### Task 2.2: Visualize Reconstructions

After training, compare original vs. reconstructed images:

```python
def visualize_reconstructions(autoencoder, x_test, n=10):
    """Show original and reconstructed images side by side."""
    reconstructed = autoencoder.predict(x_test[:n])
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## Part 3: Experiment with Bottleneck Sizes (20 min)

### Task 3.1: Compare Compression Levels

Train autoencoders with different latent dimensions and compare:

| latent_dim | Compression Ratio | Expected Quality |
|------------|-------------------|------------------|
| 64 | 784/64 = 12x | High |
| 32 | 784/32 = 24x | Good |
| 16 | 784/16 = 49x | Moderate |
| 8 | 784/8 = 98x | Low |

```python
# TODO: Train autoencoders with each latent_dim
# Record final reconstruction loss for each
# Visualize reconstructions for each
```

### Task 3.2: Latent Space Visualization

Use t-SNE to visualize the latent representations:

```python
from sklearn.manifold import TSNE

# Get latent representations
latent_representations = encoder.predict(x_test)

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_representations[:2000])

# TODO: Plot with colors for each class
# plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_test[:2000], cmap='tab10')
```

**Question:** Do similar items (e.g., sneakers and ankle boots) cluster together?

---

## Reflection Questions

1. **How does reducing latent_dim affect reconstruction quality?** At what point do images become unrecognizable?

2. **Looking at the t-SNE plot, which clothing categories cluster together?** Does this make intuitive sense?

3. **Fashion-MNIST is harder than MNIST digits. Why?** Consider the variety within each class.

4. **How could you use this autoencoder for anomaly detection?** What would happen if you fed it an image of a face?

---

## Definition of Done

- [ ] Autoencoder trained with latent_dim=32
- [ ] Reconstructions are recognizable (not random noise)
- [ ] Tested at least 3 different bottleneck sizes
- [ ] t-SNE visualization shows meaningful clusters
- [ ] Reflection questions answered with observations

---

## Bonus Challenge

Implement a **denoising autoencoder**:
1. Add random noise to input images
2. Train to reconstruct clean images
3. Test how well it removes noise from test images

```python
# Add noise
noise_factor = 0.3
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)

# Train: noisy input -> clean output
autoencoder.fit(x_train_noisy, x_train, ...)
```

---

## Solution

The solution is available in `solutions/exercise_02_solution.py`. Only check after completing your attempt!

