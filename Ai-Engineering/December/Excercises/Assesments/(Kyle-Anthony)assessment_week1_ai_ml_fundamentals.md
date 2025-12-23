# Weekly Technical Assessment: AI/ML Fundamentals

> **Week 1 Comprehensive Assessment**
> **Time Limit:** 90 minutes
> **Total Points:** 100
> **Passing Score:** 70%

---

## Section A: Conceptual Understanding (30 Points)

*Answer each question concisely. Focus on demonstrating understanding, not length.*

### A1. Machine Learning Foundations (6 points)

**A1a.** (2 pts) Define machine learning in one sentence and explain what distinguishes it from traditional programming.

<details>
<summary>ANSWER</summary>
Machine learning if the proces of designing algorithms that learn patters by analysing data to make predictions wihout being explicitly programmed.
</details>

**A1b.** (2 pts) Classify each scenario as supervised or unsupervised learning. Justify briefly.
- Scenario 1: Predicting house prices from features like bedrooms and location
- Scenario 2: Grouping news articles by topic without predefined categories

<details>
<summary>ANSWER</summary>

- **Scenario 1: Supervised Learning** - This example has would use labelled data
- **Scenario 2: Unsupervised Learning** - Data us not labelled. Trends are identified algorithmically.
</details>

**A1c.** (2 pts) Explain the difference between regression and classification with one example each.

<details>
<summary>ANSWER</summary>

- **Regression:** Predicts continuous numerical values. Example: Projected revenue ($12.3 Million)
- **Classification:** Predicts discrete categories(booleans, T/F). Example: Determining Student pass/fail 

The key is output type: continuous numbers vs discrete categories.
</details>

---

### A2. Neural Network Theory (8 points)

**A2a.** (3 pts) Draw or describe the structure of a single perceptron. Label all components and write the mathematical formula.

<details>
<summary>ANSWER</summary>

**Structure:**
1. **Inputs (x1, x2, ..., xn):** Data features fed into the perceptron
2. **Weights (w1, w2, ..., wn):** Learned values that multiply each input
3. **Bias (b):** Constant offset term
4. **Activation Function:** Non-linear function applied to the sum

**Formula:** output = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
</details>

**A2b.** (3 pts) Why are activation functions necessary in neural networks? What happens if you remove them?

<details>
<summary>ANSWER</summary>

Activation functions add non-linearity, allowing networks to learn complex patterns. Without them, the network would only be able to learn linear relationships. Even with many layers, they would collapse into a single linear transformation, losing the benefit of depth.
</details>
**A2c.** (2 pts) Compare ReLU and Sigmoid activation functions. When would you use each?

<details>
<summary>ANSWER</summary>

| Aspect | ReLU | Sigmoid |
|--------|------|---------|
| Formula | max(0, x) | 1/(1 + e^(-x)) |
| Range | [0, infinity) | (0, 1) |
| Gradient Issue | Dying ReLU | Vanishing gradients |
| Use Case | Hidden layers | Binary output layer |

- **ReLU:** Best for hidden layers - fast to compute, helps avoid vanishing gradients
- **Sigmoid:** Good for output layer in binary classification - outputs probabilities between 0 and 1
</details>

---

### A3. TensorFlow and Keras (8 points)

**A3a.** (2 pts) What is a tensor? Describe its three key properties.

<details>
<summary>ANSWER</summary>

A **tensor** is a multi-dimensional array with all elements having the same data type.

**Three Properties:**
1. **Shape:** The size of each dimension, like (32, 224, 224, 3) for batches of images
2. **Rank:** Number of dimensions (0=scalar, 1=vector, 2=matrix, etc.)
3. **Dtype:** Data type such as float32 or int64
</details>

**A3b.** (3 pts) What does the following code create? Explain each layer's purpose.

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
```

<details>
<summary>ANSWER</summary>

This builds a **Multi-Layer Perceptron for classifying into 10 categories**.

- **Layer 1 (Dense 64):** Takes 100 input features, outputs 64 values with ReLU. Learns initial patterns.
- **Layer 2 (Dense 32):** Reduces 64 to 32 values with ReLU. Learns more complex patterns.
- **Layer 3 (Dense 10):** Outputs 10 probabilities (one per class) using softmax.

Total parameters: (100*64 + 64) + (64*32 + 32) + (32*10 + 10) = 8,884
</details>

**A3c.** (3 pts) Explain what happens during `model.compile()` and why we need to call it before training.

<details>
<summary>ANSWER</summary>

`model.compile()` sets up the model for training by defining:

1. **Optimizer:** How weights get updated (like 'adam' or 'sgd')
2. **Loss Function:** How to measure errors (like 'categorical_crossentropy')
3. **Metrics:** What to display during training (like 'accuracy')

**Why needed:** The model must know how to learn (optimizer + loss) and what to show you (metrics). Without compile, `fit()` can't train because it doesn't know how to update weights or calculate gradients.
</details>

---

### A4. CNNs and Image Processing (8 points)

**A4a.** (3 pts) Explain how a convolutional layer processes an image. Include the concepts of filters, stride, and feature maps.

<details>
<summary>ANSWER</summary>

A convolutional layer uses **learnable filters** that move across the image:

1. **Filter/Kernel:** Small weight matrix (like 3x3) that detects patterns
2. **Convolution:** Multiply filter values with image patch, sum to get one output value
3. **Stride:** Step size when moving filter (stride=2 skips every other pixel)
4. **Feature Map:** Result showing where the filter detected its pattern

**Advantages:**
- Each neuron only looks at a small area (local)
- Same filter used everywhere (shared weights) - finds patterns regardless of position
- Much fewer parameters than fully connected layers
</details>

**A4b.** (2 pts) What is pooling and why is it used in CNNs?

<details>
<summary>ANSWER</summary>

**Pooling** reduces feature map size by selecting max or average from each region.

**Why use it:**
1. **Makes images smaller** - less data to process, faster training
2. **More robust to shifts** - small movements don't change the result much
3. **Keeps important features** (max pooling picks strongest signals)

Example: 2x2 max pooling: 32x32 image becomes 16x16

**Note:** Pooling doesn't learn - it's just a fixed operation.
</details>

**A4c.** (3 pts) Why is flattening necessary before connecting CNN feature maps to Dense layers?

<details>
<summary>ANSWER</summary>

**CNN produces:** 3D arrays like (height, width, channels), e.g., (8, 8, 256)

**Dense needs:** Flat 1D arrays - every neuron connects to all inputs

**Flattening:** Converts (8, 8, 256) into (16,384) - one long list

**Why needed:** Dense layers can't handle 3D data. They need everything in one dimension.

**Alternative:** GlobalAveragePooling2D averages each channel: (8, 8, 256) -> (256) - smaller but still 1D.
</details>

---

## Section B: Practical Application (40 Points)

### B1. K-Means Clustering Implementation (10 points)

Given the following 2D data points, perform K-Means clustering with K=2 for ONE iteration.

**Points:** A(1, 2), B(2, 1), C(4, 5), D(5, 4)
**Initial Centroids:** C1(1, 1), C2(5, 5)

Use Euclidean distance.

**B1a.** (4 pts) Calculate the distance from each point to each centroid.

<details>
<summary>ANSWER</summary>

**Euclidean distance:** d = sqrt((x2-x1)^2 + (y2-y1)^2)

| Point | Distance to C1(1,1) | Distance to C2(5,5) |
|-------|---------------------|---------------------|
| A(1,2) | sqrt((1-1)^2 + (2-1)^2) = sqrt(0+1) = **1.0** | sqrt((1-5)^2 + (2-5)^2) = sqrt(16+9) = **5.0** |
| B(2,1) | sqrt((2-1)^2 + (1-1)^2) = sqrt(1+0) = **1.0** | sqrt((2-5)^2 + (1-5)^2) = sqrt(9+16) = **5.0** |
| C(4,5) | sqrt((4-1)^2 + (5-1)^2) = sqrt(9+16) = **5.0** | sqrt((4-5)^2 + (5-5)^2) = sqrt(1+0) = **1.0** |
| D(5,4) | sqrt((5-1)^2 + (4-1)^2) = sqrt(16+9) = **5.0** | sqrt((5-5)^2 + (4-5)^2) = sqrt(0+1) = **1.0** |
</details>

**B1b.** (3 pts) Assign each point to the nearest centroid.

<details>
<summary>ANSWER</summary>

**Closest centroid assignment:**
- A(1,2) -> **Cluster 1** (1.0 < 5.0)
- B(2,1) -> **Cluster 1** (1.0 < 5.0)
- C(4,5) -> **Cluster 2** (1.0 < 5.0)
- D(5,4) -> **Cluster 2** (1.0 < 5.0)

**Final clusters:**
- Cluster 1: {A, B}
- Cluster 2: {C, D}
</details>

**B1c.** (3 pts) Calculate the new centroids.

<details>
<summary>ANSWER</summary>

**Centroid = average of all points in cluster**

**Cluster 1 (A, B):**
- x = (1 + 2) / 2 = **1.5**
- y = (2 + 1) / 2 = **1.5**
- **New C1 = (1.5, 1.5)**

**Cluster 2 (C, D):**
- x = (4 + 5) / 2 = **4.5**
- y = (5 + 4) / 2 = **4.5**
- **New C2 = (4.5, 4.5)**
</details>

---

### B2. Neural Network Forward Pass (15 points)

Consider a simple neural network with:
- 2 inputs
- 1 hidden layer with 2 neurons (ReLU activation)
- 1 output neuron (Sigmoid activation)

**Given weights and biases:**
```
Hidden layer:
  w1 = [0.5, 0.3]  (weights for neuron 1)
  w2 = [0.2, 0.4]  (weights for neuron 2)
  b_hidden = [0.1, -0.2]

Output layer:
  w_out = [0.6, 0.8]
  b_out = 0.3
```

**Input:** x = [2, 1]

**B2a.** (5 pts) Calculate the hidden layer output (before and after ReLU).

<details>
<summary>ANSWER</summary>

**Neuron 1 calculation:**
z1 = 0.5*2 + 0.3*1 + 0.1 = 1.0 + 0.3 + 0.1 = **1.4**
h1 = ReLU(1.4) = max(0, 1.4) = **1.4**

**Neuron 2 calculation:**
z2 = 0.2*2 + 0.4*1 + (-0.2) = 0.4 + 0.4 - 0.2 = **0.6**
h2 = ReLU(0.6) = max(0, 0.6) = **0.6**

**Hidden layer output: [1.4, 0.6]**
</details>

**B2b.** (5 pts) Calculate the final output (before and after Sigmoid).

<details>
<summary>ANSWER</summary>

**Output calculation:**
z_out = 0.6*1.4 + 0.8*0.6 + 0.3 = 0.84 + 0.48 + 0.3 = **1.62**

**Apply Sigmoid:**
y = 1 / (1 + e^(-1.62))
y = 1 / (1 + 0.198) = 1 / 1.198 = **0.835**

**Final prediction: 0.835** (about 83.5% confidence for class 1)
</details>

**B2c.** (5 pts) If the true label is 1 and we use binary cross-entropy loss, calculate the loss.

<details>
<summary>ANSWER</summary>

**Binary Cross-Entropy:** L = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]

**Values:**
- y_true = 1
- y_pred = 0.835

**Compute:**
L = -[1 * log(0.835) + 0 * log(0.165)]
L = -log(0.835)
L = -(-0.180) = **0.180**

The loss is small since 0.835 is close to the true value of 1.
</details>

---

### B3. CNN Architecture Analysis (15 points)

Analyze the following CNN architecture:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

**B3a.** (5 pts) Calculate the output shape after each layer.

<details>
<summary>ANSWER</summary>

| Layer | Input Shape | Output Shape | How |
|-------|-------------|--------------|-----|
| Input | - | (28, 28, 1) | Starting point |
| Conv2D(32) | (28, 28, 1) | **(26, 26, 32)** | 28-3+1=26, 32 filters |
| MaxPooling2D | (26, 26, 32) | **(13, 13, 32)** | 26/2=13 |
| Conv2D(64) | (13, 13, 32) | **(11, 11, 64)** | 13-3+1=11, 64 filters |
| MaxPooling2D | (11, 11, 64) | **(5, 5, 64)** | 11/2=5 |
| Flatten | (5, 5, 64) | **(1600,)** | 5*5*64=1600 |
| Dense(128) | (1600,) | **(128,)** | 128 neurons |
| Dense(10) | (128,) | **(10,)** | 10 classes |
</details>

**B3b.** (5 pts) Calculate the number of trainable parameters in the first Conv2D layer and the first Dense layer.

<details>
<summary>ANSWER</summary>

**Conv2D(32, (3,3)) with 1 input channel:**
- Weights: 3 * 3 * 1 * 32 = 288
- Biases: 32
- **Total: 320 parameters**

**Dense(128) receiving 1600 inputs:**
- Weights: 1600 * 128 = 204,800
- Biases: 128
- **Total: 204,928 parameters**

The Dense layer has about 640 times more parameters than the Conv layer, showing why convolutions are efficient for images.
</details>

**B3c.** (5 pts) What problem does this architecture solve? How can you tell from the final layer?

<details>
<summary>ANSWER</summary>

**Problem:** Classifying images into 10 categories (like recognizing digits 0-9)

**Signals:**
1. **Input (28, 28, 1):** Single-channel 28x28 images - typical for MNIST
2. **Output layer Dense(10, softmax):**
   - 10 outputs = 10 possible classes
   - Softmax gives probabilities that add up to 1.0
   - Each output is probability for one class

**Design:**
- Convolutional layers extract visual features
- Dense layers make final classification decision

The softmax with 10 outputs clearly indicates multi-class classification.
</details>

---

## Section C: Debugging and Analysis (20 Points)

### C1. Training Curve Analysis (10 points)

Given these training metrics over 20 epochs:

```
Epoch 1:  train_loss=2.5, val_loss=2.6, train_acc=0.25, val_acc=0.23
Epoch 5:  train_loss=0.8, val_loss=0.9, train_acc=0.72, val_acc=0.70
Epoch 10: train_loss=0.3, val_loss=0.7, train_acc=0.91, val_acc=0.78
Epoch 15: train_loss=0.1, val_loss=1.2, train_acc=0.98, val_acc=0.75
Epoch 20: train_loss=0.05, val_loss=1.8, train_acc=0.99, val_acc=0.72
```

**C1a.** (4 pts) Describe what is happening to the model between epochs 5 and 20.

<details>
<summary>ANSWER</summary>

**The model is overfitting starting around epoch 5:**

**Signs:**
- Training loss keeps going down (0.8 -> 0.05)
- Validation loss goes up after epoch 5 (0.9 -> 1.8)
- Training accuracy hits 99% but validation drops from 78% to 72%
- Gap between train and validation keeps widening

**What's happening:** The model memorizes training examples instead of learning useful patterns. After epoch 5, more training makes it worse on new data.
</details>

**C1b.** (3 pts) At which epoch should training have stopped? Justify your answer.

<details>
<summary>ANSWER</summary>

**Stop at epoch 5** (maybe slightly earlier between epochs 5-10).

**Reasoning:**
- Epoch 5 has smallest gap between train and validation
- Validation loss is lowest (0.9) - best performance on new data
- Validation accuracy (70%) matches training (72%) - good balance
- After epoch 5, validation loss rises - overfitting starts

**Solution:** Use EarlyStopping:
```python
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```
</details>

**C1c.** (3 pts) List three techniques to address this issue.

<details>
<summary>ANSWER</summary>

**Ways to fix overfitting:**

1. **Early Stopping:** Quit when validation stops improving
   ```python
   EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   ```

2. **Dropout:** Randomly turn off neurons during training
   ```python
   model.add(Dropout(0.5))
   ```

3. **Data Augmentation:** Generate modified versions of training images
   ```python
   ImageDataGenerator(rotation_range=20, horizontal_flip=True)
   ```

**Other options:** Add L2 regularization, use a simpler model, or get more training data.
</details>

---

### C2. Code Debugging (10 points)

Identify and fix the errors in this code:

```python
# Task: Build a model for MNIST (28x28 grayscale images, 10 classes)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='relu')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

**C2a.** (6 pts) Identify at least three errors in this code and explain why each is a problem.

<details>
<summary>ANSWER</summary>

**Error 1: Wrong input_shape**
- Should be `input_shape=(28, 28, 1)` not `(28, 28)`
- Conv2D needs 3 dimensions: height, width, channels
- Grayscale images have 1 channel

**Error 2: Missing Flatten**
- Conv2D outputs 3D data (height, width, filters)
- Dense layers need 1D input
- Must add `Flatten()` between Conv2D and Dense

**Error 3: Wrong output activation**
- Should use `activation='softmax'` not `'relu'`
- Softmax gives probabilities that sum to 1 for classification
- ReLU outputs can be any positive number, not probabilities

**Error 4: No metrics**
- Should add `metrics=['accuracy']` in compile
- Otherwise you only see loss, not accuracy during training
</details>

**C2b.** (4 pts) Write the corrected code.

<details>
<summary>ANSWER</summary>

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Fixed: added channel
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),  # Fixed: added Flatten
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Fixed: changed to softmax
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy']  # Fixed: added metrics
)

model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```
</details>

---

## Section D: Short Essay (10 Points)

Choose ONE of the following questions. Write a clear, structured response (150-250 words).

### Option 1: The Role of Feature Hierarchies in Deep Learning

Explain how neural networks learn hierarchical representations. Use a CNN processing an image of a face as a concrete example. Discuss what each layer might learn.

<details>
<summary>ANSWER</summary>

Neural networks build **hierarchical features** where each layer creates more abstract patterns from the previous layer's output.

**Example: CNN Recognizing a Face**

**Early Layers (Simple Patterns):**
First conv layers detect basic elements:
- Lines and edges in different directions
- Light and dark transitions
- Simple textures
These work for any image, not just faces.

**Middle Layers (Parts):**
Next layers combine simple features:
- Lines become curves
- Curves form shapes like eyes, nose, mouth
- Patterns create skin texture

**Deep Layers (Complex Features):**
Later conv layers recognize bigger patterns:
- Complete facial features
- How features relate to each other
- Unique characteristics for identification

**Dense Layers (Final Decision):**
Classification layers combine everything to make the final call.

**Key Point:** The network learns these patterns automatically - nobody tells it to find eyes. Each layer creates better representations for the task.
</details>

---

### Option 2: Comparing Learning Paradigms

Compare supervised and unsupervised learning from a practical perspective. Discuss when a data scientist would choose each approach, what challenges each presents, and how they might be combined.

<details>
<summary>ANSWER</summary>

**Supervised Learning** learns from labeled examples. Choose it when:
- You have clear goals (predict prices, classify emails)
- Labeled data exists or is affordable
- You can measure success against known answers

**Problems:** Labeling costs money and time, quality matters a lot, won't find unexpected patterns.

**Unsupervised Learning** finds structure without labels. Choose it when:
- Labels don't exist or cost too much
- Goal is exploration (finding groups, detecting outliers)
- You want to understand the data itself

**Problems:** Hard to know if results are good (no ground truth), need to interpret findings, might find useless patterns.

**Using Both Together:**
1. **Pre-training:** Learn general features without labels, then fine-tune with a few labels
2. **Semi-supervised:** Use clusters from unlabeled data to help with labeled examples
3. **Feature creation:** Use clustering results as inputs for supervised models

**Example:** Group customers by behavior (unsupervised), then predict purchases (supervised) using which group they're in.
</details>

---

## Answer Key Summary

| Section | Points | Passing (70%) |
|---------|--------|---------------|
| A: Conceptual | 30 | 21 |
| B: Practical | 40 | 28 |
| C: Debugging | 20 | 14 |
| D: Essay | 10 | 7 |
| **Total** | **100** | **70** |

---


<!-- Kyle-Anthony Hay -->