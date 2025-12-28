 
"""
neuron_demos.py
 
A single, demo-friendly Python file containing small, self-contained examples
based on the attached lesson markdowns:
 
07-intro-to-neural-networks.md
08-the-perceptron-model.md
09-activation-functions.md
11-forward-propagation-intuition.md
12-loss-functions-basics.md
 
HowThisFileIsMeantToBeUsed (as requested):
- Each demo is a function.
- During a live demo, call ONE function and comment out others.
- Each function prints a clear trace.
"""
 
from __future__ import annotations
 
import numpy as np
 
 
# ---------------------------------------------------------------------
# 0) Tiny helpers
# ---------------------------------------------------------------------
 
def _sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
def _relu(z):
    return np.maximum(0, z)
 
def _step(z):
    return np.where(z >= 0, 1, 0)
 
 
# ---------------------------------------------------------------------
# 1) WEIGHTS + BIAS (Attendance example) + STEP activation
# ---------------------------------------------------------------------
 
def demo_attendance_bias_and_step():
    """
    Demonstrates:
      - Inputs (evidence)
      - Weights (importance)
      - Bias (default strictness / threshold shift)
      - Step activation: PRESENT (1) vs ABSENT (0)
 
    Core formula:
      z = (w1*x1) + (w2*x2) + (w3*x3) + bias
      output = activation(z)
 
    Attendance scenario:
      x1: student is physically in class
      x2: student answered roll call
      x3: student showed ID
    """
    print("=" * 72)
    print("DEMO: Attendance -> Weights + Bias + Step Activation")
    print("=" * 72)
 
    # Evidence (inputs): 1 = yes, 0 = no
    x = np.array([1, 1, 0])  # in class, answered, ID
    print(f"Inputs (evidence) x = {x}  (in_class, answered, ID_shown)")
 
    # Importance (weights): bigger = more trusted evidence
    w = np.array([2, 2, 1])
    print(f"Weights (importance) w = {w}")
 
    # Bias: teacher's default strictness
    # bias = -2 means: need stronger evidence before marking PRESENT
    bias = -2
    print(f"Bias (strictness) b = {bias}")
 
    z = np.dot(w, x) + bias
    y = 1 if z > 0 else 0  # Step rule for demo: z>0 => PRESENT
    print(f"\nCompute confidence score:")
    print(f"  z = w·x + b = {np.dot(w, x)} + ({bias}) = {z}")
    print(f"Step activation rule: if z > 0 -> 1 (PRESENT) else 0 (ABSENT)")
    print(f"  Output y = {y}  =>  {'PRESENT ✅' if y == 1 else 'ABSENT ❌'}")
 
    # Show why bias matters with a weaker-evidence case:
    x_weak = np.array([1, 0, 0])  # only "in class"
    z_weak = np.dot(w, x_weak) + bias
    y_weak = 1 if z_weak > 0 else 0
    print("\nWhy bias matters (weak evidence case):")
    print(f"  x_weak = {x_weak} (only 'in class')")
    print(f"  z_weak = w·x_weak + b = {np.dot(w, x_weak)} + ({bias}) = {z_weak}")
    print(f"  Output y_weak = {y_weak}  =>  {'PRESENT ✅' if y_weak == 1 else 'ABSENT ✅ (filtered out weak evidence)'}")
 
 
# ---------------------------------------------------------------------
# 2) ACTIVATION FUNCTIONS (Step, Sigmoid, Tanh, ReLU, Leaky ReLU)
#    Based on "Activation Functions" file: print values at key points.
# ---------------------------------------------------------------------
 
def demo_activation_values_at_key_inputs():
    """
    Prints activation outputs at key inputs [-2, -1, 0, 1, 2]
    for Sigmoid, Tanh, ReLU, Leaky ReLU.
 
    Mirrors the 'Activation values at key inputs' table in the lesson.
    """
    print("=" * 72)
    print("DEMO: Activation function outputs at key inputs")
    print("=" * 72)
 
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
 
    def tanh_activation(z):
        return np.tanh(z)
 
    def relu(z):
        return np.maximum(0, z)
 
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
 
    keys = [-2, -1, 0, 1, 2]
    print(f"{'Input':<10} {'Sigmoid':<12} {'Tanh':<12} {'ReLU':<12} {'LeakyReLU':<12}")
    print("-" * 62)
    for val in keys:
        print(
            f"{val:<10} "
            f"{sigmoid(val):<12.4f} "
            f"{tanh_activation(val):<12.4f} "
            f"{relu(val):<12.4f} "
            f"{leaky_relu(val):<12.4f}"
        )
 
 
def demo_activation_gradients_at_key_inputs():
    """
    Prints gradient values at key inputs [-2, -1, 0, 1, 2]
    for Sigmoid, Tanh, ReLU, Leaky ReLU.
 
    Mirrors the 'Gradient values at key inputs' table in the lesson.
    """
    print("=" * 72)
    print("DEMO: Activation function gradient values at key inputs")
    print("=" * 72)
 
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
 
    def sigmoid_grad(z):
        s = sigmoid(z)
        return s * (1 - s)
 
    def tanh_grad(z):
        return 1 - np.tanh(z) ** 2
 
    def relu_grad(z):
        return np.where(z > 0, 1, 0)
 
    def leaky_relu_grad(z, alpha=0.01):
        return np.where(z > 0, 1, alpha)
 
    keys = [-2, -1, 0, 1, 2]
    print(f"{'Input':<10} {'Sigmoid':<12} {'Tanh':<12} {'ReLU':<12} {'LeakyReLU':<12}")
    print("-" * 62)
    for val in keys:
        print(
            f"{val:<10} "
            f"{sigmoid_grad(val):<12.4f} "
            f"{tanh_grad(val):<12.4f} "
            f"{relu_grad(val):<12.4f} "
            f"{leaky_relu_grad(val):<12.4f}"
        )
 
 
# ---------------------------------------------------------------------
# 3) INTRO FILE: SimpleNeuron, DenseLayer, SimpleNetwork (forward passes)
# ---------------------------------------------------------------------
 
class SimpleNeuron:
    """A single artificial neuron (from the intro lesson)."""
 
    def __init__(self, num_inputs: int):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
 
    def forward(self, inputs: np.ndarray) -> float:
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = 1 / (1 + np.exp(-weighted_sum))  # sigmoid
        return float(output)
 
 
def demo_single_neuron_forward():
    """
    Runs the conceptual demonstration from the intro reading:
    - random weights + bias
    - compute weighted sum
    - sigmoid activation
    """
    print("=" * 72)
    print("DEMO: Single Neuron forward pass (Intro lesson)")
    print("=" * 72)
 
    neuron = SimpleNeuron(num_inputs=3)
    inputs = np.array([0.5, 0.3, 0.2])
    output = neuron.forward(inputs)
 
    print(f"Inputs:  {inputs}")
    print(f"Weights: {neuron.weights}")
    print(f"Bias:    {neuron.bias:.3f}")
    print(f"Output (sigmoid): {output:.3f}")
 
    # show the raw z too for clarity
    z = np.dot(inputs, neuron.weights) + neuron.bias
    print(f"z = w·x + b = {z:.3f}")
 
 
class DenseLayer:
    """A layer of multiple neurons (from the intro lesson)."""
 
    def __init__(self, num_inputs: int, num_neurons: int):
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.1
        self.biases = np.zeros(num_neurons)
 
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        z = np.dot(inputs, self.weights) + self.biases
        return 1 / (1 + np.exp(-z))  # sigmoid
 
 
def demo_dense_layer_forward():
    """
    Demonstrates:
      - vector input -> vector output
      - matrix weights -> multiple neurons at once
    """
    print("=" * 72)
    print("DEMO: Dense layer forward pass (Intro lesson)")
    print("=" * 72)
 
    layer = DenseLayer(num_inputs=3, num_neurons=4)
    inputs = np.array([0.5, 0.3, 0.2])
    outputs = layer.forward(inputs)
 
    print(f"Input shape:  {inputs.shape}  input={inputs}")
    print(f"Weights shape:{layer.weights.shape}  (num_inputs x num_neurons)")
    print(f"Biases shape: {layer.biases.shape}")
    print(f"Output shape: {outputs.shape}  outputs={outputs}")
 
 
class SimpleNetwork:
    """A minimal feedforward network (Intro lesson): 3->4->2"""
 
    def __init__(self):
        self.layer1 = DenseLayer(num_inputs=3, num_neurons=4)
        self.layer2 = DenseLayer(num_inputs=4, num_neurons=2)
 
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = self.layer1.forward(inputs)
        x = self.layer2.forward(x)
        return x
 
 
def demo_simple_network_forward():
    """
    Demonstrates stacking layers:
      input -> hidden -> output
    """
    print("=" * 72)
    print("DEMO: Simple network forward pass (Intro lesson)")
    print("=" * 72)
 
    network = SimpleNetwork()
    inputs = np.array([0.5, 0.3, 0.2])
    outputs = network.forward(inputs)
 
    print(f"Network input:  {inputs}")
    print(f"Network output: {outputs}")
    print(f"Predicted class (argmax): {int(np.argmax(outputs))}")
 
 
# ---------------------------------------------------------------------
# 4) PERCEPTRON MODEL (AND gate training) from the perceptron file
# ---------------------------------------------------------------------
 
class Perceptron:
    """A single perceptron (binary classifier) from the perceptron lesson."""
 
    def __init__(self, num_features: int, learning_rate: float = 0.1):
        self.weights = np.random.randn(num_features) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
 
    def activation(self, z):
        return np.where(z >= 0, 1, 0)
 
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
 
    def train(self, X, y, epochs: int = 100):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi.reshape(1, -1))[0]
                error = yi - prediction
                if error != 0:
                    errors += 1
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break
        return self
 
 
def demo_perceptron_and_gate():
    """
    Trains a perceptron on the AND truth table and prints results.
    (This comes directly from the perceptron lesson's code example.)
    """
    print("=" * 72)
    print("DEMO: Perceptron training on AND gate")
    print("=" * 72)
 
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])
 
    perceptron = Perceptron(num_features=2, learning_rate=0.1)
    perceptron.train(X, y, epochs=100)
 
    print("\nAND Gate Results:")
    print(f"Weights: {perceptron.weights}")
    print(f"Bias:    {perceptron.bias:.3f}")
    for xi, yi in zip(X, y):
        pred = perceptron.predict(xi.reshape(1, -1))[0]
        print(f"Input: {xi} -> Predicted: {pred}, Actual: {yi}")
 
 
# ---------------------------------------------------------------------
# 5) FORWARD PROPAGATION INTUITION (numerical trace) from forward-prop file
# ---------------------------------------------------------------------
 
def demo_forward_propagation_trace():
    """
    Reproduces the forward propagation numerical example:
      Input: [0.5, 0.8]
      Hidden layer (ReLU): 2 -> 3
      Output layer (Sigmoid): 3 -> 1
 
    Prints Z and A at each layer.
    """
    print("=" * 72)
    print("DEMO: Forward propagation trace (2 -> 3 -> 1)")
    print("=" * 72)
 
    X = np.array([[0.5, 0.8]])  # shape (1,2)
 
    parameters = {
        'W1': np.array([[0.2, -0.3, 0.4],
                        [0.5,  0.1, -0.2]]),
        'b1': np.array([[0.1, -0.1, 0.2]]),
        'W2': np.array([[0.6],
                        [-0.4],
                        [0.3]]),
        'b2': np.array([[0.1]])
    }
 
    # Layer 1
    Z1 = X @ parameters['W1'] + parameters['b1']
    A1 = np.maximum(0, Z1)  # ReLU
 
    # Layer 2
    Z2 = A1 @ parameters['W2'] + parameters['b2']
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid
 
    print(f"Input X:   {X}")
    print(f"Z1:        {Z1}")
    print(f"A1 (ReLU): {A1}")
    print(f"Z2:        {Z2}")
    print(f"A2 (Sigmoid output): {A2}")
    print(f"Interpretation: ~{float(A2[0,0])*100:.1f}% probability of positive class")
 
 
# ---------------------------------------------------------------------
# 6) LOSS FUNCTIONS (MSE + BCE) from the loss-functions file
# ---------------------------------------------------------------------
 
def demo_loss_functions_mse_and_bce():
    """
    Demonstrates:
      - Mean Squared Error (MSE) for regression
      - Binary Cross-Entropy (BCE) for binary classification
 
    Prints the values and per-sample contributions.
    """
    print("=" * 72)
    print("DEMO: Loss functions (MSE and Binary Cross-Entropy)")
    print("=" * 72)
 
    # --- Regression (MSE) ---
    print("\n--- Regression: Mean Squared Error (MSE) ---")
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.5, 5.5, 6.0, 10.0])
 
    mse = np.mean((y_true - y_pred) ** 2)
    errors = y_true - y_pred
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"errors: {errors}")
    print(f"squared errors: {errors**2}")
    print(f"MSE: {mse:.4f}")
 
    # --- Binary classification (BCE) ---
    print("\n--- Binary Classification: Binary Cross-Entropy (BCE) ---")
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.7, 0.2, 0.4])  # probabilities
 
    epsilon = 1e-15
    y_pred_safe = np.clip(y_pred, epsilon, 1 - epsilon)
 
    bce = -np.mean(
        y_true * np.log(y_pred_safe) +
        (1 - y_true) * np.log(1 - y_pred_safe)
    )
 
    print(f"y_true: {y_true}")
    print(f"y_pred(prob): {y_pred}")
    print(f"BCE: {bce:.4f}")
 
    print("\nPer-sample loss contributions:")
    for yt, yp in zip(y_true, y_pred_safe):
        if yt == 1:
            sample_loss = -np.log(yp)
        else:
            sample_loss = -np.log(1 - yp)
        print(f"  y={yt}, p={float(yp):.2f} -> loss={float(sample_loss):.4f}")
 
 
# ---------------------------------------------------------------------
# Optional: A simple "menu" function for convenience
# ---------------------------------------------------------------------
 
def list_available_demos():
    """
    Prints the demo functions you can call during class.
    """
    demos = [
        "demo_attendance_bias_and_step()",
        "demo_activation_values_at_key_inputs()",
        "demo_activation_gradients_at_key_inputs()",
        "demo_single_neuron_forward()",
        "demo_dense_layer_forward()",
        "demo_simple_network_forward()",
        "demo_perceptron_and_gate()",
        "demo_forward_propagation_trace()",
        "demo_loss_functions_mse_and_bce()",
    ]
    print("Available demo functions:")
    for d in demos:
        print("  -", d)
 
 
if __name__ == "__main__":
    # Uncomment ONE demo at a time while presenting.
    #list_available_demos()
 
    # demo_attendance_bias_and_step()
    # demo_activation_values_at_key_inputs()
    # demo_activation_gradients_at_key_inputs()
    # demo_single_neuron_forward()
    # demo_dense_layer_forward()
    demo_simple_network_forward()
    # demo_perceptron_and_gate()
    # demo_forward_propagation_trace()
    # demo_loss_functions_mse_and_bce()
 
 