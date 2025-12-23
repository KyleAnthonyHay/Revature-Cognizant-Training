# Exercise 03: CNN Architecture Tuning with TensorBoard

## Learning Objectives

- Experiment with CNN hyperparameters (filters, kernel sizes, pooling)
- Use descriptive TensorBoard log names for easy comparison
- Overlay multiple training runs in TensorBoard to visualize trade-offs
- Make data-driven architecture decisions

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_03_mnist_mlp_classifier.py`, you saw a 2-block CNN achieve 99%+ accuracy. But how sensitive is CNN performance to architecture choices? In this exercise, you'll train multiple variations and compare them side-by-side in TensorBoard.

## Your Challenge

Train 4-6 CNN variations with **descriptive log names** so you can overlay them in TensorBoard and make data-driven decisions about architecture.

---

## Part 1: Setup and Baseline (10 min)

### Task 1.1: Create the Training Function

Open `starter_code/cnn_tuning_starter.py` and implement a flexible training function:

```python
def train_cnn(
    filters=[32, 64],       # Number of filters per conv block
    kernel_size=3,          # Kernel size (3 or 5)
    pool_size=2,            # MaxPool size
    dense_units=64,         # Dense layer neurons
    experiment_name="baseline"  # TensorBoard log name!
):
    """Train a CNN and log to TensorBoard with descriptive name."""
    
    # Build model based on parameters
    model = keras.Sequential([...])
    
    # IMPORTANT: Use experiment_name for log directory
    log_dir = f"logs/cnn_tuning/{experiment_name}"
    
    # Train and return history
    ...
```

### Task 1.2: Train Baseline

Train the baseline configuration and note the log directory name:

```python
train_cnn(
    filters=[32, 64],
    kernel_size=3,
    pool_size=2,
    dense_units=64,
    experiment_name="baseline_32-64_k3"  # Descriptive name!
)
```

---

## Part 2: Filter Experiments (15 min)

### Task 2.1: Vary Number of Filters

Train these configurations with descriptive names:

| Experiment | Filters | Log Name |
|------------|---------|----------|
| Fewer filters | [16, 32] | `filters_16-32_k3` |
| Baseline | [32, 64] | `baseline_32-64_k3` |
| More filters | [64, 128] | `filters_64-128_k3` |

### Task 2.2: Compare in TensorBoard

1. Run: `tensorboard --logdir=logs/cnn_tuning`
2. In the SCALARS tab, you should see all 3 runs overlaid
3. Use the smoothing slider to see clearer trends

**Record your observations:**
- Which configuration converges fastest?
- Which has the lowest final validation loss?
- Is the accuracy difference worth the extra parameters?

---

## Part 3: Kernel Size Experiments (15 min)

### Task 3.1: Vary Kernel Size

Keep filters at [32, 64], vary kernel:

| Experiment | Kernel | Log Name |
|------------|--------|----------|
| Small kernel | 3 | `baseline_32-64_k3` (already done) |
| Large kernel | 5 | `kernel_32-64_k5` |

### Task 3.2: Observe in TensorBoard

Refresh TensorBoard and compare:
- Does 5x5 kernel capture more features?
- How does training time compare?
- Is there more overfitting with larger kernels?

---

## Part 4: Pooling and Dense Experiments (10 min)

### Task 4.1: Vary Pool Size

| Experiment | Pool Size | Log Name |
|------------|-----------|----------|
| 2x2 pooling | 2 | `baseline_32-64_k3` |
| 3x3 pooling | 3 | `pool3_32-64_k3` |

### Task 4.2: Vary Dense Units

| Experiment | Dense Units | Log Name |
|------------|-------------|----------|
| Small dense | 32 | `dense32_32-64_k3` |
| Baseline | 64 | `baseline_32-64_k3` |
| Large dense | 128 | `dense128_32-64_k3` |

---

## Part 5: Analysis and Optimal Model (10 min)

### Task 5.1: TensorBoard Analysis

Look at your runs overlaid in TensorBoard and answer:

1. **Best filter configuration:** _______________
2. **Best kernel size:** _______________  
3. **Best dense layer size:** _______________
4. **Signs of overfitting (if any):** _______________

### Task 5.2: Train Optimal Model

Combine your best findings into one final model:

```python
train_cnn(
    filters=[...],        # Your best
    kernel_size=...,      # Your best
    pool_size=2,
    dense_units=...,      # Your best
    experiment_name="optimal_final"
)
```

---

## TensorBoard Tips

**Overlay runs:** All runs in the same parent directory (`logs/cnn_tuning/`) appear together

**Filter runs:** Use the search box to show only specific experiments, e.g., type "filters" to see only filter experiments

**Smoothing:** Adjust the smoothing slider (0.6 works well) to see clearer trends

**Download:** You can download scalar data as CSV for further analysis

---

## Definition of Done

- [ ] At least 3 filter configurations trained and compared
- [ ] At least 2 kernel sizes compared
- [ ] TensorBoard shows all runs overlaid
- [ ] Observations recorded for each experiment
- [ ] Optimal configuration identified and trained
- [ ] Final test accuracy achieved: > 98.5%

---

## Reflection Questions

1. **Which hyperparameter had the biggest impact on accuracy?**

2. **Did more filters always mean better accuracy?** What's the trade-off?

3. **How did you use TensorBoard to make your decisions?** What patterns did you look for?

4. **If you had to deploy this model to a mobile device, which configuration would you choose and why?**

---

## Solution

The solution is available in `solutions/exercise_03_solution.py`. Only check after completing your attempt!
