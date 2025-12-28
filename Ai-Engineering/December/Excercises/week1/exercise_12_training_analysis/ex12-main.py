"""
Exercise 12: Filter Explorer Lab

HOW TO RUN:
1. Install required dependencies:
   pip install -r requirements.txt
   
   Or install individually:
   pip install numpy matplotlib tensorflow

2. Run the script:
   python3 ex12-main.py

NOTE: Multiple matplotlib windows will open. Close each window to proceed
      to the next visualization, or run in a non-interactive environment.
"""
import matplotlib.pyplot as plt

# Part 2: Training Histories
history_a = {
    'loss':     [0.8, 0.5, 0.3, 0.15, 0.08, 0.03, 0.01, 0.005],
    'val_loss': [0.9, 0.6, 0.5, 0.55, 0.65, 0.80, 0.95, 1.10],
    'accuracy':     [0.70, 0.82, 0.90, 0.95, 0.98, 0.99, 0.998, 0.999],
    'val_accuracy': [0.68, 0.78, 0.82, 0.81, 0.79, 0.77, 0.75, 0.73]
}

history_b = {
    'loss':     [0.9, 0.85, 0.82, 0.80, 0.79, 0.78, 0.77, 0.76],
    'val_loss': [0.95, 0.90, 0.88, 0.86, 0.85, 0.84, 0.83, 0.82],
    'accuracy':     [0.30, 0.35, 0.38, 0.40, 0.41, 0.42, 0.43, 0.44],
    'val_accuracy': [0.28, 0.33, 0.36, 0.38, 0.39, 0.40, 0.41, 0.42]
}

history_c = {
    'loss':     [0.8, 0.3, 0.9, 0.2, 1.1, 0.15, 1.5, 0.1],
    'val_loss': [0.9, 0.5, 1.2, 0.4, 1.8, 0.3, 2.5, 0.25],
    'accuracy':     [0.65, 0.85, 0.60, 0.90, 0.55, 0.92, 0.50, 0.95],
    'val_accuracy': [0.60, 0.78, 0.52, 0.82, 0.45, 0.85, 0.40, 0.88]
}

history_d = {
    'loss':     [2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30],
    'val_loss': [2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30],
    'accuracy':     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    'val_accuracy': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
}

history_e = {
    'loss':     [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    'val_loss': [0.55, 0.25, 0.15, 0.10, 0.08, 0.07, 0.065, 0.062],
    'accuracy':     [0.85, 0.93, 0.97, 0.99, 0.995, 0.998, 0.999, 0.9995],
    'val_accuracy': [0.82, 0.90, 0.94, 0.96, 0.97, 0.975, 0.978, 0.979]
}

# Part 2: Analysis Functions
def analyze_case_1(history):
    print("=" * 60)
    print("Case 1: The Overfit Model")
    print("=" * 60)
    print("\n1. What is happening? (Diagnosis)")
    print("   Answer: Overfitting - Training loss decreases while validation")
    print("           loss increases after epoch 3. Model memorizes training data.")
    
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    print(f"\n2. At which epoch should training have stopped?")
    print(f"   Answer: Epoch {best_epoch} (lowest validation loss)")
    
    print("\n3. What remedies would you recommend? (List at least 3)")
    print("   - Remedy 1: Add dropout layers")
    print("   - Remedy 2: Reduce model complexity")
    print("   - Remedy 3: Use early stopping")
    print()

def analyze_case_2(history):
    print("=" * 60)
    print("Case 2: The Underfit Model")
    print("=" * 60)
    print("\n1. What is happening? (Diagnosis)")
    print("   Answer: Underfitting - Both losses high and improving slowly.")
    print("           Model too simple to learn patterns.")
    
    print("\n2. Why are training and validation metrics similar but both poor?")
    print("   Answer: Model lacks capacity to learn, so it fails equally")
    print("           on both sets. Not overfitting, just insufficient learning.")
    
    print("\n3. What remedies would you recommend? (List at least 3)")
    print("   - Remedy 1: Increase model capacity (more layers/neurons)")
    print("   - Remedy 2: Train for more epochs")
    print("   - Remedy 3: Reduce learning rate for better convergence")
    print()

def analyze_case_3(history):
    print("=" * 60)
    print("Case 3: The Unstable Model")
    print("=" * 60)
    print("\n1. What is happening? (Diagnosis)")
    print("   Answer: Unstable training - Loss and accuracy, the two metrics are oscillating wildly.")
    print("           Also, the model is not converging smoothly.")
    
    print("\n2. What is causing the oscillation?")
    print("   Answer: Learning rate too high, causing optimizer to overshoot")
    print("           optimal weights and jump around loss landscape.")
    
    print("\n3. What remedies would you recommend?")
    print("   - Remedy 1: Reduce learning rate")
    print("   - Remedy 2: Use learning rate scheduling")
    print("   - Remedy 3: Increase batch size")
    print()

def analyze_case_4(history):
    print("=" * 60)
    print("Case 4: The Stuck Model")
    print("=" * 60)
    print("\n1. What is happening? (Diagnosis)")
    print("   Answer: Model completely stuck - no learning occurring.")
    print("           All metrics remain constant across epochs.")
    
    print("\n2. The accuracy is exactly 10% - what does this suggest for a 10-class problem?")
    print("   Answer: Model predicting randomly or always same class.")
    print("           Random guessing gives ~10% for 10 classes.")
    
    print("\n3. What could cause the model to be completely stuck?")
    print("   - Possible cause 1: Learning rate is zero or too small")
    print("   - Possible cause 2: Gradients not flowing (vanishing gradients)")
    print("   - Possible cause 3: Weights frozen or not being updated")
    print()

def analyze_case_5(history):
    print("=" * 60)
    print("Case 5: The Almost Perfect Model")
    print("=" * 60)
    print("\n1. Is this model overfitting? How can you tell?")
    print("   Answer: Slight overfitting but acceptable. Training accuracy")
    print("           99.95% vs validation 97.9% (~2% gap). Both improving.")
    
    gap = history['accuracy'][-1] - history['val_accuracy'][-1]
    print(f"\n2. What is the gap between training and validation accuracy?")
    print(f"   Answer: {gap:.3f} ({history['accuracy'][-1]:.4f} - {history['val_accuracy'][-1]:.4f})")
    
    print("\n3. Is this acceptable? Should anything be done?")
    print("   Answer: Yes, acceptable. Model performing well. Optional:")
    print("           monitor for divergence, consider slight regularization.")
    print()

    
# ========================================================
# Part 3: Visualize the Cases
# ========================================================
def plot_training_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0].plot(epochs, history['loss'], 'b-', label='Training', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['accuracy'], 'b-', label='Training', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ========================================================
# Part 4: Remedies Reference
# ========================================================
def print_remedies_table():
    print("=" * 80)
    print("Part 4: Remedies Reference Table")
    print("=" * 80)
    print()
    print(f"{'Problem':<25} {'Symptom':<30} {'Remedies'}")
    print("-" * 80)
    print(f"{'Overfitting':<25} {'Train loss << Val loss':<30} {'1. Add dropout  2. Reduce complexity  3. Early stopping'}")
    print(f"{'Underfitting':<25} {'Both losses high':<30} {'1. Increase capacity  2. More epochs  3. Lower LR'}")
    print(f"{'Unstable training':<25} {'Oscillating metrics':<30} {'1. Reduce learning rate  2. LR scheduling'}")
    print(f"{'Stuck training':<25} {'No improvement':<30} {'1. Check LR  2. Check gradients  3. Verify weights'}")
    print(f"{'Slow convergence':<25} {'Very gradual improvement':<30} {'1. Increase LR  2. Better initialization'}")
    print()

# ========================================================
# Part 5: Real Debugging Scenario
# ========================================================
def print_debugging_scenario():
    print("=" * 80)
    print("Part 5: Real Debugging Scenario")
    print("=" * 80)
    print()
    print("Error Report:")
    print("  Model: CNN for image classification")
    print("  Dataset: 50,000 training images, 10,000 test images")
    print("  Training: 50 epochs")
    print("  Result: Training accuracy 99.9%, Test accuracy 65%")
    print()
    print("Debugging Plan:")
    print()
    print("1. First thing to check:")
    print("   Plot training vs validation curves to confirm overfitting")
    print("   Verify data preprocessing consistency between train/test")
    print()
    print("2. Likely diagnosis:")
    print("   Severe overfitting - model memorized training data")
    print()
    print("3. Code changes to try (be specific):")
    print("   - Add Dropout: model.add(Dropout(0.5)) or maybe 0.25 after conv/dense layers")
    print("   - Add L2 regularization: Dense(128, kernel_regularizer=l2(0.001))")
    print()
    print("4. How to prevent this in the future:")
    print("   - Monitor validation metrics during training")
    print("   - Implement early stopping callback")
    print("   - Start with simpler models, gradually increase complexity")
    print("   - Use validation set")
    print()

if __name__ == '__main__':
    analyze_case_1(history_a)
    analyze_case_2(history_b)
    analyze_case_3(history_c)
    analyze_case_4(history_d)
    analyze_case_5(history_e)
    
    plot_training_history(history_a, 'Case 1: Overfitting')
    plot_training_history(history_b, 'Case 2: Underfitting')
    plot_training_history(history_c, 'Case 3: Unstable Training')
    plot_training_history(history_d, 'Case 4: Stuck Model')
    plot_training_history(history_e, 'Case 5: Almost Perfect Model')
    
    print_remedies_table()
    print_debugging_scenario()

