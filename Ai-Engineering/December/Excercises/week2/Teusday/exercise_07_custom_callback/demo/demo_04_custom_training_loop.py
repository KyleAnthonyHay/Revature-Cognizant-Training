"""
Demo 04: Custom Training Loops

This demo shows trainees how to:
1. Build custom training loops with GradientTape
2. Implement manual gradient computation
3. Add custom logging and callbacks
4. Gain full control over training process

Learning Objectives:
- Understand what model.fit() does under the hood
- Learn to write custom training loops
- Apply custom callbacks and learning rate schedules

References:
- Written Content: 05-customizing-training-process.md
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os

# ============================================================================
# PART 1: Understanding model.fit() Under the Hood
# ============================================================================

print("=" * 70)
print("PART 1: What model.fit() Does Behind the Scenes")
print("=" * 70)

print("\nmodel.fit() automates:")
print("-" * 40)
print("""
1. Create batches from training data
2. For each batch:
   a. Forward pass: compute predictions
   b. Compute loss: compare to targets
   c. Backward pass: compute gradients
   d. Update weights: apply optimizer
3. Log metrics
4. Validate on validation set
5. Repeat for N epochs
""")

print("Sometimes you need more control. Let's build our own!")

# ============================================================================
# PART 2: Basic Custom Training Loop
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Basic Custom Training Loop")
print("=" * 70)

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Use subset
x_train_sub = x_train[:10000]
y_train_sub = y_train[:10000]

# Create model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Define loss and optimizer
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Define metrics
train_loss_metric = keras.metrics.Mean(name='train_loss')
train_acc_metric = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

print("Model, loss function, and optimizer defined")
print(f"Loss: {loss_fn.name}")
print(f"Optimizer: {optimizer.__class__.__name__}")

# Single training step function
@tf.function  # Compile to graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)
        # Compute loss
        loss = loss_fn(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y, predictions)
    
    return loss

# Create dataset
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Set up TensorBoard logging for custom training
log_dir = "logs/custom_training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
file_writer = tf.summary.create_file_writer(log_dir)

# Training loop
epochs = 10
history = {'loss': [], 'accuracy': []}

print(f"\nTensorBoard logs: {log_dir}")
print("View with: tensorboard --logdir=logs/custom_training")
print(f"Starting custom training loop for {epochs} epochs...")
print("-" * 50)

for epoch in range(epochs):
    # Reset metrics at start of epoch
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()
    
    # Iterate over batches
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
    
    # Get epoch metrics
    epoch_loss = train_loss_metric.result().numpy()
    epoch_acc = train_acc_metric.result().numpy()
    
    history['loss'].append(epoch_loss)
    history['accuracy'].append(epoch_acc)
    
    # Log to TensorBoard
    with file_writer.as_default():
        tf.summary.scalar('loss', epoch_loss, step=epoch)
        tf.summary.scalar('accuracy', epoch_acc, step=epoch)
        file_writer.flush()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

print("-" * 50)
print("Custom training complete!")

# ============================================================================
# PART 3: Custom Training with Validation
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Adding Validation to Custom Loop")
print("=" * 70)

# Fresh model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Train metrics
train_loss_metric = keras.metrics.Mean()
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Validation metrics
val_loss_metric = keras.metrics.Mean()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y, predictions)

@tf.function
def val_step(x, y):
    predictions = model(x, training=False)  # No dropout during validation
    loss = loss_fn(y, predictions)
    
    val_loss_metric.update_state(loss)
    val_acc_metric.update_state(y, predictions)

# Split data
x_train_final = x_train_sub[:8000]
y_train_final = y_train_sub[:8000]
x_val = x_train_sub[8000:]
y_val = y_train_sub[8000:]

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_final, y_train_final))
train_dataset = train_dataset.shuffle(1024).batch(128)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(128)

# Training with validation
epochs = 10
full_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

print(f"\nTraining with validation for {epochs} epochs...")
print("-" * 50)

for epoch in range(epochs):
    # Reset all metrics
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()
    val_loss_metric.reset_state()
    val_acc_metric.reset_state()
    
    # Training
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    # Validation
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)
    
    # Record metrics
    full_history['loss'].append(train_loss_metric.result().numpy())
    full_history['accuracy'].append(train_acc_metric.result().numpy())
    full_history['val_loss'].append(val_loss_metric.result().numpy())
    full_history['val_accuracy'].append(val_acc_metric.result().numpy())
    
    print(f"Epoch {epoch+1}/{epochs} - "
          f"Loss: {full_history['loss'][-1]:.4f} - "
          f"Acc: {full_history['accuracy'][-1]:.4f} - "
          f"Val_Loss: {full_history['val_loss'][-1]:.4f} - "
          f"Val_Acc: {full_history['val_accuracy'][-1]:.4f}")

# ============================================================================
# PART 4: Custom Learning Rate Schedule
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Custom Learning Rate Schedule")
print("=" * 70)

# Fresh model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Custom learning rate schedule
class CustomLRSchedule:
    def __init__(self, initial_lr=0.01, decay_rate=0.9, decay_steps=1000):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step = 0
        self.lr_history = []
    
    def get_lr(self):
        lr = self.initial_lr * (self.decay_rate ** (self.step / self.decay_steps))
        self.step += 1
        self.lr_history.append(lr)
        return lr

lr_schedule = CustomLRSchedule(initial_lr=0.01, decay_rate=0.95, decay_steps=100)

# Use SGD optimizer (manual LR update)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

train_loss_metric = keras.metrics.Mean()
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

def train_step_with_custom_lr(x, y):
    # Get current learning rate
    current_lr = lr_schedule.get_lr()
    optimizer = keras.optimizers.SGD(learning_rate=current_lr)
    
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y, predictions)

print("Training with custom exponential decay learning rate...")
print("-" * 50)

epochs = 10
lr_history = {'loss': [], 'accuracy': [], 'lr': []}

for epoch in range(epochs):
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()
    
    for x_batch, y_batch in train_dataset:
        train_step_with_custom_lr(x_batch, y_batch)
    
    current_lr = lr_schedule.lr_history[-1]
    lr_history['loss'].append(train_loss_metric.result().numpy())
    lr_history['accuracy'].append(train_acc_metric.result().numpy())
    lr_history['lr'].append(current_lr)
    
    print(f"Epoch {epoch+1}/{epochs} - "
          f"Loss: {lr_history['loss'][-1]:.4f} - "
          f"Acc: {lr_history['accuracy'][-1]:.4f} - "
          f"LR: {current_lr:.6f}")

# ============================================================================
# PART 5: Gradient Clipping
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Gradient Clipping")
print("=" * 70)

# Fresh model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step_with_clipping(x, y, clip_norm=1.0):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Clip gradients by global norm
    gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
    
    # Apply clipped gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, global_norm

print("Demonstrating gradient clipping...")
print("-" * 50)

for step, (x_batch, y_batch) in enumerate(train_dataset.take(5)):
    loss, grad_norm = train_step_with_clipping(x_batch, y_batch, clip_norm=1.0)
    print(f"Step {step+1}: Loss={loss.numpy():.4f}, Gradient Norm (after clipping)={grad_norm.numpy():.4f}")

print("\nGradient clipping prevents exploding gradients by capping the norm")
print("Useful for RNNs and very deep networks")

# ============================================================================
# PART 6: Visualize Training Results
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizing Training Results")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
ax = axes[0, 0]
ax.plot(full_history['loss'], label='Train', linewidth=2)
ax.plot(full_history['val_loss'], label='Validation', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Accuracy
ax = axes[0, 1]
ax.plot(full_history['accuracy'], label='Train', linewidth=2)
ax.plot(full_history['val_accuracy'], label='Validation', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training and Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Learning Rate Schedule
ax = axes[1, 0]
ax.plot(lr_schedule.lr_history, linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Learning Rate')
ax.set_title('Custom Learning Rate Schedule')
ax.grid(True, alpha=0.3)

# Learning Rate vs Loss
ax = axes[1, 1]
ax.plot(lr_history['lr'], lr_history['loss'], 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Loss')
ax.set_title('Loss vs Learning Rate')
ax.grid(True, alpha=0.3)

plt.suptitle('Custom Training Loop Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('custom_training_results.png', dpi=150)
print("[OK] Training visualization saved to: custom_training_results.png")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Custom Training Loops")
print("=" * 70)

print("\nKey Takeaways:")
print("1. GradientTape records operations for automatic differentiation")
print("2. tape.gradient() computes gradients")
print("3. optimizer.apply_gradients() updates weights")
print("4. @tf.function compiles to graph for ~10x speedup")
print("5. Custom loops enable: gradient clipping, custom LR schedules, complex architectures")
print("6. Always separate training and validation metrics")

print("\nWhen to use custom training loops:")
print("- GAN training (alternate discriminator/generator)")
print("- Multi-task learning")
print("- Reinforcement learning")
print("- Custom gradient modifications")
print("- Advanced debugging")

print("\n" + "=" * 70)

