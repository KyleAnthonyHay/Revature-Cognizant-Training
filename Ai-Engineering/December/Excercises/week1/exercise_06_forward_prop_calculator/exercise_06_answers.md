# Exercise 06: Forward Propagation Calculator - Answers

## Part 1: Paper Calculations

### Step 1: Hidden Layer Linear Transform

```
z1[0] = X[0]*W1[0,0] + X[1]*W1[1,0] + X[2]*W1[2,0] + b1[0]
z1[0] = 1.0*0.2 + 0.5*0.3 + (-0.5)*(-0.1) + 0.1
z1[0] = 0.2 + 0.15 + 0.05 + 0.1
z1[0] = 0.50

z1[1] = X[0]*W1[0,1] + X[1]*W1[1,1] + X[2]*W1[2,1] + b1[1]
z1[1] = 1.0*(-0.5) + 0.5*0.4 + (-0.5)*0.2 + (-0.1)
z1[1] = -0.5 + 0.2 + -0.1 + (-0.1)
z1[1] = -0.50
```

**Your answer:** z1 = [0.50, -0.50]

### Step 2: Hidden Layer Activation (ReLU)

```
a1[0] = max(0, z1[0]) = max(0, 0.50) = 0.50
a1[1] = max(0, z1[1]) = max(0, -0.50) = 0.00
```

**Your answer:** a1 = [0.50, 0.00]

### Step 3: Output Layer Linear Transform

```
z2[0] = a1[0]*W2[0,0] + a1[1]*W2[1,0] + b2[0]
z2[0] = 0.50*0.6 + 0.00*(-0.3) + 0.1
z2[0] = 0.30 + 0.00 + 0.1
z2[0] = 0.40
```

**Your answer:** z2 = [0.40]

### Step 4: Output Layer Activation (Sigmoid)

```
y_hat = 1 / (1 + exp(-0.40))
y_hat = 1 / (1 + 0.6703)
y_hat = 1 / 1.6703
y_hat = 0.5987
```

**Your answer:** y_hat = 0.5987

### Step 5: Interpret the Output

```
The network predicts: 0.5987 (probability of positive class)

If threshold = 0.5:
Classification: 1 (0 or 1)
Note: At this threshold its a lean towards true but not very confident.
```

---

## Part 3: Different Inputs

### Input 2: X = [0.0, 1.0, 1.0]

```
z1 = [ 0.5 -0.5]
a1 = [0.5 0. ]
z2 = [0.4]
y_hat = [0.59868766]
```

### Input 3: X = [-1.0, 0.0, 0.0]

```
z1 = [-0.1  0.4]
a1 = [0.  0.4]
z2 = [-0.02]
y_hat = [0.49500017]
```

### Input 4: X = [2.0, 2.0, 2.0]

```
z1 = [0.9 0.1]
a1 = [0.9 0.1]
z2 = [0.61]
y_hat = [0.6479408]
```

### Analysis

```python
# Q1: Which input(s) produced y_hat > 0.5 (positive class)?
# Answer: Input 1 (0.5987), Input 2 (0.5987), and Input 4 (0.6479) all produced y_hat > 0.5.
#         Input 3 (0.4950) is below the threshold, so it would be classified as negative class.

# Q2: For Input 3, what happened at the ReLU layer? Why?
# Answer: The first neuron's output was set to 0.0 because z1[0] = -0.1 was negative, 
#         and ReLU sets all negative values to zero. This is the "dying ReLU" effect.
#         Only the second neuron (a1[1] = 0.4) contributed to the output layer calculation.

# Q3: How does the network's output change with different inputs?
# Answer: The network's output varies based on the weighted combination of inputs through 
#         the layers. Input 1 and Input 2 both produce similar outputs (0.5987), showing 
#         the network can map different input patterns to similar predictions. Input 3 
#         produces the lowest output (0.4950) due to negative input values being "killed" 
#         by ReLU, while Input 4 produces the highest output (0.6479) due to larger 
#         positive input values. The sigmoid function maps the final linear combination 
#         to a probability between 0 and 1, with values above 0.5 indicating positive class.
```

