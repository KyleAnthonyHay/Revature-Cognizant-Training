# Exercise 02: Optimizer Comparison

## Learning Objectives

- Compare convergence behavior of different optimizers
- Understand the impact of learning rate on training
- Visualize optimization trajectories
- Select appropriate optimizers for different scenarios

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_02_gradient_descent_comparison.py`, you saw how Adam converges faster than vanilla SGD. In this exercise, you'll systematically compare optimizers and understand when to use each.

## Optimizer Reference

```
# Vanilla SGD: Simple gradient descent
# w = w - lr * gradient

# SGD with Momentum: Accumulates velocity
# v = momentum * v - lr * gradient
# w = w + v

# RMSprop: Adapts learning rate per parameter
# cache = decay * cache + (1-decay) * gradient^2
# w = w - lr * gradient / sqrt(cache + epsilon)

# Adam: Combines momentum and adaptive learning rates
# m = beta1 * m + (1-beta1) * gradient       # First moment
# v = beta2 * v + (1-beta2) * gradient^2     # Second moment
# w = w - lr * m / sqrt(v + epsilon)
```

---

## Part 1: Setup Experiment (10 min)

### Task 1.1: Create Consistent Training Environment

Navigate to `starter_code/optimizer_comparison_starter.py`. Set up an experiment that ensures fair comparison:

```python
def create_dataset():
    """Load and prepare MNIST for consistent comparison."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_model(optimizer):
    """
    Create identical model architecture for each optimizer.
    
    Use same:
    - Random seed for weight initialization
    - Architecture: [128, 64, 10]
    - Activation: ReLU for hidden, Softmax for output
    """
    # Set seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # TODO: Build model with specified optimizer
    pass
```

---

## Part 2: Compare Optimizers (20 min)

### Task 2.1: Train with Each Optimizer

Run experiments with the following configurations:

| Optimizer | Configuration |
|-----------|---------------|
| SGD | `lr=0.01` |
| SGD + Momentum | `lr=0.01, momentum=0.9` |
| RMSprop | `lr=0.001` |
| Adam | `lr=0.001` |

For each optimizer:
1. Train for 20 epochs
2. Record training loss and validation accuracy per epoch
3. Store histories for comparison

### Task 2.2: Visualize Convergence

Create a plot with:
- X-axis: Epoch
- Y-axis (left): Training loss
- Y-axis (right): Validation accuracy
- One line per optimizer

Include a legend and title.

---

## Part 3: Learning Rate Sensitivity (15 min)

### Task 3.1: Test Learning Rate Extremes

Using Adam optimizer, test:
- `lr=0.1` (too high)
- `lr=0.01` (high)
- `lr=0.001` (default)
- `lr=0.0001` (low)
- `lr=0.00001` (too low)

Record and plot the training curves.

### Task 3.2: Find Optimal Learning Rate

Implement a simple learning rate finder:

```python
def find_learning_rate(model, x_train, y_train, start_lr=1e-7, end_lr=1):
    """
    Exponentially increase learning rate and track loss.
    
    The optimal LR is typically 1/10th of where loss starts exploding.
    """
    # TODO: Implement LR finder
    # 1. Start with very small LR
    # 2. Train for one batch
    # 3. Increase LR exponentially  
    # 4. Plot loss vs LR
    pass
```

---

## Analysis Questions

Complete these as part of your code or in comments:

1. **Which optimizer converged fastest to 95% accuracy?**

2. **Did any optimizer fail to converge with the default settings?**

3. **What happened when learning rate was too high (0.1)?** Describe the training curve behavior.

4. **Based on your results, which optimizer would you recommend for:**
   - Quick prototyping?
   - Final training run for best accuracy?
   - Limited computational budget?

---

## Results Summary

Complete this table:

| Optimizer | Epochs to 95% Acc | Final Val Acc | Training Time |
|-----------|-------------------|---------------|---------------|
| SGD | | | |
| SGD+Momentum | | | |
| RMSprop | | | |
| Adam | | | |

---

## Reflection Questions

1. **Why does momentum help SGD converge faster?** Think about the optimization landscape.

2. **Adam is often the default choice. When might you choose SGD instead?**

3. **How did the learning rate finder help identify a good starting point?**

4. **If training loss decreases but validation accuracy plateaus, what does this indicate?**

---

## Definition of Done

- [ ] All four optimizers compared with identical models
- [ ] Convergence plot created with all optimizers
- [ ] Learning rate sensitivity experiment completed
- [ ] Results summary table filled in
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_02_solution.py`. Only check after completing your attempt!

