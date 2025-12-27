# Pair Programming Analysis: RNN vs LSTM

## Results
| Model     | Val Accuracy | Training Time |
|-----------|--------------|---------------|
| SimpleRNN | 82.4%        | 18 seconds    |
| LSTM      | 85.0%        | 30 seconds    |

## Analysis

### 1. Which model achieved higher accuracy and by how much?
LSTM achieved ~2.6% higher validation accuracy (85.0% vs 82.4%).

### 2. Training time comparison - which was faster and why?
SimpleRNN was ~2x faster because LSTM has more parameters.

### 3. Early convergence - which learned faster in first 2 epochs?
LSTM—reached 85% val accuracy by epoch 1, while SimpleRNN was at 79%.

### 4. Why does LSTM work better for long reviews?
LSTM is good at remembering long-term dependencies.

### 5. When would SimpleRNN be the better choice?
on short sequences or when speed is mroe improtant than accuracy.

## Key Takeaway
LSTM's gating mechanism trades training time for better long-sequence accuracy.
