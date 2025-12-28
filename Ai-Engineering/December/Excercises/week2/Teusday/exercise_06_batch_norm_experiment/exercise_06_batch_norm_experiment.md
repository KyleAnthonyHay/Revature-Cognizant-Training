# Exercise 03: Batch Normalization Experiment

## Learning Objectives

- Implement batch normalization in deep networks
- Measure the impact on training speed and stability
- Understand when and where to place batch norm layers
- Experiment with higher learning rates enabled by batch normalization

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_03_batch_norm_impact.py`, you saw how batch normalization accelerates training. Now you'll design experiments to quantify these benefits and understand the mechanics.

## Key Concepts

```
# Batch Normalization Formula:
# For a mini-batch B of size m:

# 1. Compute mean: mu_B = (1/m) * sum(x_i)
# 2. Compute variance: sigma_B^2 = (1/m) * sum((x_i - mu_B)^2)
# 3. Normalize: x_hat = (x - mu_B) / sqrt(sigma_B^2 + epsilon)
# 4. Scale and shift: y = gamma * x_hat + beta

# gamma and beta are learnable parameters!
```

---

## Part 1: Build Comparison Models (15 min)

### Task 1.1: Model Without Batch Normalization

Create a deep network (4+ hidden layers) without batch normalization.

Architecture:
- Input: 784
- Hidden layers: [256, 128, 64, 32]
- Output: 10 (softmax)

### Task 1.2: Model With Batch Normalization

Create the same architecture with batch normalization after each hidden layer.

Best practice placement:
```python
# Option A: BN after activation (common)
layers.Dense(256, activation='relu')
layers.BatchNormalization()

# Option B: BN before activation (also valid)  
layers.Dense(256, use_bias=False)  # Note: use_bias=False when BN follows
layers.BatchNormalization()
layers.Activation('relu')
```

---

## Part 2: Convergence Speed Comparison (20 min)

### Task 2.1: Train Both Models

Train both models with:
- Same optimizer (Adam, lr=0.001)
- Same epochs (30)
- Same batch size (128)
- Same random seed

Record per-epoch metrics.

### Task 2.2: Plot Comparison

Create a visualization showing:
- Training loss curves (both models)
- Validation accuracy curves (both models)
- Mark the epoch where each reaches 95% accuracy

### Task 2.3: Calculate Speedup

Compute:
- Epochs to 95% accuracy (with BN) / Epochs to 95% accuracy (without BN)
- Final accuracy difference

---

## Part 3: Learning Rate Tolerance (15 min)

### Task 3.1: High Learning Rate Experiment

One benefit of batch normalization is tolerance to higher learning rates.

Experiment with learning rates that would normally cause training to diverge:

| Learning Rate | Without BN | With BN |
|---------------|------------|---------|
| 0.001 | (baseline) | (baseline) |
| 0.01 | | |
| 0.05 | | |
| 0.1 | | |

For each combination, record:
- Did training converge? (Yes/No)
- Final validation accuracy (if converged)

### Task 3.2: Find Maximum Stable Learning Rate

For each model, find the highest learning rate that still converges.

---

## Part 4: Inspect Batch Norm Parameters (10 min)

### Task 4.1: Examine Learned Parameters

After training, inspect the batch normalization layers:

```python
for layer in model.layers:
    if isinstance(layer, layers.BatchNormalization):
        gamma = layer.gamma.numpy()
        beta = layer.beta.numpy()
        moving_mean = layer.moving_mean.numpy()
        moving_var = layer.moving_variance.numpy()
        
        print(f"Layer: {layer.name}")
        print(f"  gamma: mean={gamma.mean():.3f}, std={gamma.std():.3f}")
        print(f"  beta: mean={beta.mean():.3f}, std={beta.std():.3f}")
        print(f"  moving_mean: {moving_mean.mean():.3f}")
        print(f"  moving_var: {moving_var.mean():.3f}")
```

Questions to answer:
- Do gamma values stay close to 1.0?
- Do beta values stay close to 0.0?
- What do the moving statistics tell you about the activations?

---

## Results Summary

### Convergence Comparison

| Metric | Without BN | With BN | Improvement |
|--------|------------|---------|-------------|
| Epochs to 95% acc | | | |
| Final val accuracy | | | |
| Training time/epoch | | | |

### Learning Rate Tolerance

| Learning Rate | Without BN Converges? | With BN Converges? |
|---------------|----------------------|-------------------|
| 0.001 | | |
| 0.01 | | |
| 0.05 | | |
| 0.1 | | |

---

## Reflection Questions

1. **By how many epochs did batch normalization speed up training?** Was this consistent across different learning rates?

2. **What is the maximum stable learning rate with and without batch normalization?** How much higher can you go with BN?

3. **Looking at the learned gamma and beta values, did any layers learn to "undo" the normalization?** (gamma far from 1 or beta far from 0)

4. **Would you always use batch normalization? Are there any downsides?** Consider inference time and batch size requirements.

---

## Definition of Done

- [ ] Both models (with/without BN) implemented
- [ ] Convergence comparison completed and plotted
- [ ] Learning rate tolerance experiment completed
- [ ] BN parameters inspected and analyzed
- [ ] Results tables filled in
- [ ] Reflection questions answered

---

## Bonus Challenge

Implement Layer Normalization and compare it to Batch Normalization:

```python
layers.LayerNormalization()  # Normalizes across features instead of batch
```

When might you prefer Layer Normalization over Batch Normalization?

---

## Solution

The solution is available in `solutions/exercise_03_solution.py`. Only check after completing your attempt!

