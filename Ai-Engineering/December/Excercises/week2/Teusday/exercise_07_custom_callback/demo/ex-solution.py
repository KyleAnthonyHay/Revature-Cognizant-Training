"""
Exercise 04 Solution: Custom Callbacks

Complete implementation of custom Keras callbacks.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import os

os.makedirs('logs/exercise_callbacks', exist_ok=True)

# =============================================================================
# TASK 1: Step Learning Rate Scheduler
# =============================================================================

class StepLRScheduler(keras.callbacks.Callback):
    """
    Custom callback that reduces learning rate by a factor every N epochs.
    
    Example:
        scheduler = StepLRScheduler(factor=0.5, step_epochs=10)
        # LR will be halved at epochs 10, 20, 30, etc.
    """
    
    def __init__(self, factor=0.5, step_epochs=10, verbose=True):
        super().__init__()
        self.factor = factor
        self.step_epochs = step_epochs
        self.verbose = verbose
        self.lr_history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """Adjust learning rate at the beginning of each epoch."""
        # Get current learning rate (Keras 3 compatible)
        lr_var = self.model.optimizer.learning_rate
        if hasattr(lr_var, 'numpy'):
            current_lr = float(lr_var.numpy())
        else:
            current_lr = float(lr_var)
        
        # Check if we should reduce LR
        if epoch > 0 and epoch % self.step_epochs == 0:
            new_lr = current_lr * self.factor
            # Keras 3 compatible way to set learning rate
            self.model.optimizer.learning_rate.assign(new_lr)
            
            if self.verbose:
                print(f"\nEpoch {epoch}: Reducing LR from {current_lr:.6f} to {new_lr:.6f}")
            
            current_lr = new_lr
        
        # Track LR history
        self.lr_history.append(current_lr)
    
    def get_lr_history(self):
        """Return the learning rate history."""
        return self.lr_history


# =============================================================================
# TASK 2: JSON Training Logger
# =============================================================================

class JSONTrainingLogger(keras.callbacks.Callback):
    """
    Custom callback that logs training metrics to a JSON file.
    
    Output format:
    [
        {"epoch": 1, "timestamp": "...", "loss": 0.5, "accuracy": 0.8, ...},
        {"epoch": 2, "timestamp": "...", "loss": 0.4, "accuracy": 0.85, ...},
    ]
    """
    
    def __init__(self, log_file='training_log.json'):
        super().__init__()
        self.log_file = log_file
        self.logs_list = []
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        """Initialize the log file at training start."""
        self.logs_list = []
        self.start_time = datetime.datetime.now()
        print(f"Training logger initialized. Logging to: {self.log_file}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics after each epoch."""
        logs = logs or {}
        
        # Create log entry
        entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0))
        }
        self.logs_list.append(entry)
        
        # Write to file (overwrite with full list each time for valid JSON)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs_list, f, indent=2)
    
    def on_train_end(self, logs=None):
        """Print summary at training end."""
        total_time = datetime.datetime.now() - self.start_time
        print(f"\nTraining complete!")
        print(f"Total time: {total_time}")
        print(f"Logged {len(self.logs_list)} epochs to {self.log_file}")


# =============================================================================
# TASK 3: Combined Training
# =============================================================================

def create_model():
    """Create simple model for testing callbacks."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),  # Start with high LR
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_with_custom_callbacks():
    """Train model using both custom callbacks."""
    print("\n" + "=" * 50)
    print("Task 3: Training with Custom Callbacks")
    print("=" * 50)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Create model
    model = create_model()
    
    # Create callbacks
    lr_scheduler = StepLRScheduler(factor=0.5, step_epochs=10)
    json_logger = JSONTrainingLogger('training_log.json')
    
    # Add early stopping for comparison
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add TensorBoard to show how to combine with custom callbacks
    log_dir = "logs/exercise_callbacks/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"TensorBoard logs: {log_dir}")
    
    # Train
    history = model.fit(
        x_train[:10000], y_train[:10000],
        epochs=30,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[lr_scheduler, json_logger, early_stop, tb_callback],
        verbose=1
    )
    
    # Verify LR schedule
    print("\nLearning Rate History:")
    for i, lr in enumerate(lr_scheduler.get_lr_history()):
        if i % 10 == 0 or i == len(lr_scheduler.get_lr_history()) - 1:
            print(f"  Epoch {i}: LR = {lr:.6f}")
    
    # Verify JSON log
    with open('training_log.json', 'r') as f:
        logs = json.load(f)
    print(f"\nJSON log contains {len(logs)} entries")
    print(f"First entry: {json.dumps(logs[0], indent=2)}")
    print(f"Last entry: {json.dumps(logs[-1], indent=2)}")
    
    return history, lr_scheduler.get_lr_history()


def visualize_from_json(log_file='training_log.json'):
    """Create visualizations from the JSON log file."""
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    epochs = [log['epoch'] for log in logs]
    train_loss = [log['loss'] for log in logs]
    val_loss = [log['val_loss'] for log in logs]
    train_acc = [log['accuracy'] for log in logs]
    val_acc = [log['val_accuracy'] for log in logs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_loss, 'b-', label='Train')
    axes[0].plot(epochs, val_loss, 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss (from JSON log)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_acc, 'b-', label='Train')
    axes[1].plot(epochs, val_acc, 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy (from JSON log)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_from_json.png', dpi=150)
    plt.show()
    print("Saved visualization to 'training_from_json.png'")


# =============================================================================
# BONUS: Warmup + Cosine Decay Scheduler
# =============================================================================

class WarmupCosineScheduler(keras.callbacks.Callback):
    """
    Learning rate schedule with:
    1. Linear warmup for first N epochs
    2. Cosine decay for remaining epochs
    """
    
    def __init__(self, base_lr=0.01, warmup_epochs=5, total_epochs=50):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.lr_history.append(lr)
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}: LR = {lr:.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 04 Solution: Custom Callbacks")
    print("=" * 60)
    
    # Train with custom callbacks
    history, lr_history = train_with_custom_callbacks()
    
    # Visualize from JSON
    print("\n" + "=" * 60)
    print("Visualizing from JSON log")
    print("=" * 60)
    visualize_from_json()
    
    # Plot LR schedule
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Step Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_schedule.png', dpi=150)
    plt.show()
    print("Saved LR schedule to 'lr_schedule.png'")
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)
