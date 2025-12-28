# Exercise 04: Custom Callbacks

## Learning Objectives

- Understand the Keras callback system and lifecycle methods
- Create custom callbacks for learning rate scheduling
- Implement training loggers for metric tracking
- Combine multiple callbacks in a training pipeline

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_04_custom_training_loop.py`, you saw how to take control of the training process. This exercise focuses on creating reusable custom callbacks that integrate with `model.fit()` while providing advanced functionality.

## Key Concepts

```
# Keras Callback Lifecycle:
# 
# on_train_begin(logs)        <- Called once at training start
#   |
#   +-> on_epoch_begin(epoch, logs)   <- Start of each epoch
#   |     |
#   |     +-> on_batch_begin(batch, logs)
#   |     |     |
#   |     |     +-> [Training step]
#   |     |     |
#   |     +-> on_batch_end(batch, logs)
#   |     |
#   +-> on_epoch_end(epoch, logs)     <- End of each epoch
#   |
# on_train_end(logs)          <- Called once at training end
```

---

## Part 1: Step Learning Rate Scheduler (15 min)

### Task 1.1: Implement StepLRScheduler

Create a callback that reduces the learning rate by a factor every N epochs.

Open `starter_code/custom_callback_starter.py` and complete the `StepLRScheduler` class:

```python
class StepLRScheduler(keras.callbacks.Callback):
    """
    Reduces learning rate by `factor` every `step_epochs` epochs.
    
    Example:
        scheduler = StepLRScheduler(factor=0.5, step_epochs=10)
        # LR: 0.1 -> 0.05 (epoch 10) -> 0.025 (epoch 20) -> ...
    """
    
    def __init__(self, factor=0.5, step_epochs=10, verbose=True):
        super().__init__()
        # TODO: Store parameters
        # TODO: Initialize lr_history list
    
    def on_epoch_begin(self, epoch, logs=None):
        # TODO: Get current learning rate from self.model.optimizer
        # TODO: If epoch is multiple of step_epochs, reduce LR
        # TODO: Set new LR using keras.backend.set_value()
        # TODO: Track LR in history
        pass
```

### Task 1.2: Test the Scheduler

```python
# Test your scheduler
model = create_simple_model()

scheduler = StepLRScheduler(factor=0.5, step_epochs=5)

history = model.fit(
    x_train, y_train,
    epochs=20,
    callbacks=[scheduler],
    verbose=0
)

# Verify LR changes
lr_history = scheduler.get_lr_history()
print("Learning Rate History:")
for i, lr in enumerate(lr_history):
    print(f"  Epoch {i}: {lr:.6f}")
```

Expected output:
```
Epoch 5: Reducing LR from 0.100000 to 0.050000
Epoch 10: Reducing LR from 0.050000 to 0.025000
Epoch 15: Reducing LR from 0.025000 to 0.012500
```

---

## Part 2: JSON Training Logger (15 min)

### Task 2.1: Implement JSONTrainingLogger

Create a callback that logs all training metrics to a JSON file:

```python
class JSONTrainingLogger(keras.callbacks.Callback):
    """
    Logs training metrics to a JSON file after each epoch.
    
    Output format:
    [
        {"epoch": 1, "timestamp": "...", "loss": 0.5, "accuracy": 0.8, ...},
        {"epoch": 2, "timestamp": "...", "loss": 0.4, "accuracy": 0.85, ...},
    ]
    """
    
    def __init__(self, log_file='training_log.json'):
        super().__init__()
        # TODO: Store log_file path
        # TODO: Initialize logs_list
    
    def on_train_begin(self, logs=None):
        # TODO: Initialize/clear log file
        # TODO: Store start time
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        # TODO: Create entry with epoch, timestamp, and all metrics
        # TODO: Append to logs_list
        # TODO: Write updated list to JSON file
        pass
    
    def on_train_end(self, logs=None):
        # TODO: Calculate total training time
        # TODO: Print summary
        pass
```

### Task 2.2: Test the Logger

```python
# Test your logger
model = create_simple_model()

logger = JSONTrainingLogger('my_training_log.json')

history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[logger],
    verbose=0
)

# Verify JSON output
import json
with open('my_training_log.json', 'r') as f:
    logs = json.load(f)

print(f"Logged {len(logs)} epochs")
print(f"First entry: {json.dumps(logs[0], indent=2)}")
```

---

## Part 3: Combined Training Pipeline (15 min)

### Task 3.1: Use Multiple Callbacks Together

Combine your custom callbacks with built-in ones:

```python
def train_with_full_pipeline():
    """
    Train with multiple callbacks working together.
    """
    model = create_simple_model()
    
    # Your custom callbacks
    lr_scheduler = StepLRScheduler(factor=0.5, step_epochs=10)
    json_logger = JSONTrainingLogger('experiment_log.json')
    
    # Built-in callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # All callbacks together
    callbacks = [
        lr_scheduler,
        json_logger,
        early_stop
    ]
    
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history, lr_scheduler.get_lr_history()
```

### Task 3.2: Analyze the Results

After training, analyze:

1. How many epochs did training run before early stopping?
2. What was the final learning rate?
3. Load the JSON log and plot metrics over time

```python
# Plot from JSON log
import matplotlib.pyplot as plt

with open('experiment_log.json', 'r') as f:
    logs = json.load(f)

epochs = [log['epoch'] for log in logs]
train_loss = [log['loss'] for log in logs]
val_loss = [log['val_loss'] for log in logs]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress (from JSON log)')
plt.savefig('training_from_json.png')
plt.show()
```

---

## Bonus Challenge: Warmup + Decay Scheduler

Implement a more sophisticated learning rate schedule:

```python
class WarmupCosineScheduler(keras.callbacks.Callback):
    """
    Learning rate schedule with:
    1. Linear warmup for first N epochs
    2. Cosine decay for remaining epochs
    
    LR = base_lr * warmup_factor  (during warmup)
    LR = base_lr * 0.5 * (1 + cos(pi * progress))  (during decay)
    """
    
    def __init__(self, base_lr=0.01, warmup_epochs=5, total_epochs=50):
        super().__init__()
        # TODO: Implement warmup + cosine decay
        pass
```

---

## Results Summary

Complete this table:

| Callback | Function | Verified Working? |
|----------|----------|-------------------|
| StepLRScheduler | Reduce LR every N epochs | |
| JSONTrainingLogger | Log metrics to JSON | |
| Combined Pipeline | All callbacks together | |

---

## Reflection Questions

1. **When would you use a custom callback vs. a built-in one?** What functionality is missing from built-in callbacks?

2. **In what order are callbacks executed when you pass multiple?** Does the order in the list matter?

3. **How would you save your custom callback for reuse across projects?** (Think about packaging and imports)

4. **What happens if a callback raises an exception during training?** How would you make your callbacks robust?

---

## Definition of Done

- [ ] StepLRScheduler implemented and tested
- [ ] JSONTrainingLogger implemented and tested
- [ ] Combined pipeline runs successfully
- [ ] JSON log file generated and readable
- [ ] Training visualization created from JSON log
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_04_solution.py`. Only check after completing your attempt!

