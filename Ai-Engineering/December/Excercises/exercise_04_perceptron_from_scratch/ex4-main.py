import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    #===============================================
    # Task 1.1: Class Skeleton
    #===============================================
    """A single perceptron (binary classifier)."""
    
    def __init__(self, n_features, learning_rate=0.1, activation='step'):
        """
        Initialize the perceptron.
        
        Args:
            n_features: Number of input features
            learning_rate: Step size for weight updates
            activation: 'step' or 'sigmoid'
        """
        # TODO: Initialize weights with small random values
        self.weights = np.random.randn(n_features) * 0.01  # Shape: (n_features,)
        
        # TODO: Initialize bias to 0
        self.bias = 0
        
        self.learning_rate = learning_rate
        self.activation_name = activation
        
    def _activation(self, z):
        """
        Apply activation function.
        
        Args:
            z: Weighted sum (can be scalar or array)
        
        Returns:
            Activated output
        """
        if self.activation_name == 'step':
            # TODO: Return 1 if z >= 0, else 0
            return np.where(z >= 0, 1, 0)
        elif self.activation_name == 'sigmoid':
            # TODO: Return 1 / (1 + exp(-z))
            return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """
        Compute forward pass.
        
        Args:
            X: Input features (n_samples, n_features) or (n_features,)
        
        Returns:
            Predictions
        """
        # TODO: Compute z = X @ weights + bias
        z = np.dot(X, self.weights) + self.bias
        # TODO: Apply activation
        return self._activation(z)
    
    def predict(self, X):
        """Make predictions (alias for forward)."""
        output = self.forward(X)
        if self.activation_name == 'sigmoid':
            return (output >= 0.5).astype(int)
        return output.astype(int)
    
    def fit(self, X, y, epochs=100):
        """
        Train the perceptron.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            epochs: Number of passes through data
        
        Returns:
            self, history
        """
        history = {'errors': [], 'weights': [], 'bias': []}
        
        for epoch in range(epochs):
            errors = 0
            
            for xi, yi in zip(X, y):
                # TODO: Get prediction 
                prediction = self.predict(xi.reshape(1, -1))[0]
                
                # TODO: Calculate error
                error = yi - prediction
                
                # TODO: Update weights and bias if error != 0
                if error != 0:
                    # w = w + lr * error * x
                    # b = b + lr * error
                    errors += 1
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
            
            # Record history
            history['errors'].append(errors)
            history['weights'].append(self.weights.copy())
            history['bias'].append(self.bias)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Errors = {errors}")
            
            # Early stopping if no errors
            if errors == 0:
                print(f"Converged at epoch {epoch}!")
                break
        
        return self, history

    #==============================================================================================
    # Task 2.1: Plot Decision Boundary
    #==============================================================================================
    def plot_decision_boundary(self, X, y, title):
        """
        Visualize the decision boundary.
        
        The decision boundary is where: w1*x1 + w2*x2 + b = 0
        Solving for x2: x2 = -(w1*x1 + b) / w2
        """
        
        plt.figure(figsize=(8, 6))
        
        # TODO: Plot data points
        # Red circles for class 0, green squares for class 1
        for label, marker, color in [(0, 'o', 'red'), (1, 's', 'green')]:
            mask = y == label 
            plt.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, 
                    s=150, label=f'Class {label}', edgecolors='black', linewidth=2)
        # TODO: Plot decision boundary line
        # x1_range = np.linspace(-0.5, 1.5, 100)
        # x2_boundary = -(w1 * x1_range + b) / w2
        w1, w2 = perceptron.weights
        b = perceptron.bias
        x1_range = np.linspace(-0.5, 1.5, 100)
        x2_boundary = -(w1 * x1_range + b) / w2
        plt.plot(x1_range, x2_boundary, 'b-', linewidth=2, label='Decision Boundary')
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    

#==============================================================================================
# Task 1.2: Test with AND Gate
#==============================================================================================
#  main function
if __name__ == "__main__":
    #===============================================
    # AND gate data
    #===============================================
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])

    # TODO: Create and train perceptron
    perceptron = Perceptron(n_features=2, learning_rate=0.1)
    perceptron, history = perceptron.fit(X_and, y_and, epochs=100)

    # TODO: Test predictions
    predictions = perceptron.predict(X_and)
    print(f"Predictions: {predictions}")
    print(f"Actual: {y_and}")
    print(f"Accuracy: {(predictions == y_and).mean():.2%}")

    #===============================================
    # Task 2.2: Learning Curve
    #===============================================
    # TODO: Plot errors over epochs
    plt.plot(history['errors'])
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Errors')
    plt.grid(True, alpha=0.3)
    plt.show()

    #===============================================
    # OR gate data
    #===============================================
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    # TODO: Create and train new perceptron for OR
    perceptron_or = Perceptron(n_features=2, learning_rate=0.1)
    perceptron_or, history_or = perceptron_or.fit(X_or, y_or, epochs=100)

    # TODO: Test predictions
    predictions_or = perceptron_or.predict(X_or)
    print("================================================")
    print(f"Predictions: {predictions_or}")
    print(f"Actual: {y_or}")
    print(f"Accuracy: {(predictions_or == y_or).mean():.2%}")
    
    #  TODO: NEXT
    #===============================================
    # Task 3.1: XOR gate data 
    #===============================================


    # TODO: Call for decision boundary plot for AND and OR gates
    perceptron.plot_decision_boundary(X_and, y_and, 'AND Gate: Perceptron Decision Boundary')
    perceptron_or.plot_decision_boundary(X_or, y_or, 'OR Gate: Perceptron Decision Boundary')