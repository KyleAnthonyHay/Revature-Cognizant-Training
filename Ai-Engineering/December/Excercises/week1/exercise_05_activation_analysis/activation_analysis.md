# Activation Function Analysis Report

**Name:** ____Kyle-Anthony Hay___________  
**Date:** ____December 18, 2025___________

---

## 1. Output Range Comparison

| Activation | Range | Best For |
|------------|-------|----------|
| Step       |{0, 1} |Hard threshold decisions          |
| Sigmoid    |(0, 1) |Binary classification output layer |
| Tanh       |(-1, 1)|RNNs, zero-centered hidden layers  |
| ReLU       |[0, ∞) |Default for hidden layers         |
| Leaky ReLU |(-∞, ∞)|Hidden layers (prevents dead neurons)|

---

## 2. Gradient Behavior

### Sigmoid

- At z = 0: gradient = 0.250000
- At z = 5: gradient = 0.006648
- At z = -5: gradient = 0.006648
- Problem identified: Vanishing gradient at extremes (z=±5), gradients become very small

### Tanh

- At z = 0: gradient = 1.000000
- At z = 5: gradient = 0.000181
- Problem identified: Vanishing gradient at extremes, though better than sigmoid (zero-centered)

### ReLU

- At z > 0: gradient = 1.0
- At z < 0: gradient = 0.0
- Problem identified: Dying ReLU - neurons with z < 0 have zero gradient and stop learning

### Leaky ReLU

- At z > 0: gradient = 1.0
- At z < 0: gradient = 0.01 (alpha)
- How it addresses ReLU's problem: Small positive gradient (0.01) for negative inputs allows weight updates, preventing dead neurons

---

## 3. Vanishing Gradient Analysis

For sigmoid at z = 10:
- Output = 0.9999
- Gradient = 0.000045

Explanation of why this is problematic:
When the gradient is extremely small(near zero), weight updates during backpropagation become negligible. In deep networks, this problem compounds across layers - early layers receive tiny gradients and learn very slowly or stop learning entirely. 

This is why sigmoid and tanh are problematic for deep networks.



---

## 4. Dead ReLU Analysis

What causes neurons to "die"?

When a ReLU neuron receives negative inputs (z < 0), it outputs 0 and has a gradient of 0. If the weights consistently produce negative z values, the neuron stays at 0 forever. With zero gradient, the weights cannot update, so the neuron becomes permanently "dead" and contributes nothing to the network's learning capacity.

How does Leaky ReLU fix this?

Leaky ReLU fixes this by usin a small positive gradient (alpha = 0.01) for negative inputs instead of zero. This allows weights to update even when z < 0, preventing neurons from dying completely. The small gradient ensures neurons can recover and contribute to learning.



---

## 5. Recommendations

### For Hidden Layers

I would use: ReLU or Leaky ReLU

Reasons:
1. No vanishing gradient problem for positive values - constant gradient of 1
2. Computationally efficient - simple max(0, z) operation
3. Helps with sparse activations and faster convergence

### For Output Layer (Binary Classification)

I would use: Sigmoid

Reasons:
1. Outputs probabilities in range [0, 1] - perfect for binary classification
2. Smooth, differentiable function suitable for gradient-based learning
3. Interpretable as probability of positive class

### For Output Layer (Multi-Class Classification)

I would use: Softmax

Reason:
Outputs probability distribution over all classes (sums to 1), allowing selection of most likely class

---

## 6. Key Insight

The most important thing I learned about activation functions is:

Activation functions are crucial for introducing non-linearity into neural networks, but their gradient behavior directly impacts learning. The vanishing gradient problem (sigmoid/tanh) and dying ReLU problem show that choosing the right activation function is essential for deep network training. ReLU became the default for hidden layers because it solves vanishing gradients, while sigmoid/softmax remain essential for output layers where probability interpretation is needed.




---

## 7. Visualization Sketch

Draw rough sketches of each activation function:

```
Sigmoid:          Tanh:             ReLU:
   |  ___            |   ___           |    /
   | /               |  /              |   /
---|/---         ---|----          ---|----
   |                __|              /  |
   |                                /   |
```

