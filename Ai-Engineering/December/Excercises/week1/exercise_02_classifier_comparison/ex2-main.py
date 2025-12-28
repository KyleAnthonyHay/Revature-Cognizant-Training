"""
Exercise 02: Classifier Comparison Lab
======================================

Part 1: Data Preparation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print("=" * 60)
print("CLASSIFIER COMPARISON LAB - PART 1: DATA PREPARATION")
print("=" * 60)

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# Task 1.1: Print dataset info
print("\n--- DATASET INFO ---")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {data.feature_names}")
print(f"Class distribution: {np.bincount(y)}")
print(f"  - Malignant (0): {np.bincount(y)[0]} samples")
print(f"  - Benign (1): {np.bincount(y)[1]} samples")

# Task 1.2: Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n--- TRAIN/TEST SPLIT ---")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Task 1.2: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n--- FEATURE SCALING ---")
print("Features scaled using StandardScaler")
print(f"Training set mean (first feature): {X_train_scaled[:, 0].mean():.4f}")
print(f"Training set std (first feature): {X_train_scaled[:, 0].std():.4f}")

print("\n" + "=" * 60)
print("Part 1 Complete: Data is ready for training!")
print("=" * 60)

# =============================================================================
# PART 2: TRAIN MULTIPLE CLASSIFIERS
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: TRAIN MULTIPLE CLASSIFIERS")
print("=" * 60)

# Task 2.1: Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42)
}

print("\n--- TRAINING CLASSIFIERS ---")

# Task 2.2: Train and evaluate each classifier
results = {}
trained_classifiers = {}

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    trained_classifiers[name] = clf
    
    y_pred = clf.predict(X_test_scaled)
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print(f"{name}: Accuracy = {results[name]['accuracy']:.2%}")

print("\n" + "=" * 60)
print("Part 2 Complete: All classifiers trained and evaluated!")
print("=" * 60)

# =============================================================================
# PART 3: DETAILED ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: DETAILED ANALYSIS")
print("=" * 60)

# Task 3.1: Create comparison table
print("\n--- COMPARISON TABLE ---")
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)
print(comparison_df)

# Find best and worst classifier based on F1 score
best_clf_name = comparison_df['f1'].idxmax()
worst_clf_name = comparison_df['f1'].idxmin()
print(f"\nBest Classifier: {best_clf_name} (F1 = {comparison_df.loc[best_clf_name, 'f1']:.4f})")
print(f"Worst Classifier: {worst_clf_name} (F1 = {comparison_df.loc[worst_clf_name, 'f1']:.4f})")

# Task 3.2: Visualize comparison
print("\n--- VISUALIZATION ---")
plt.figure(figsize=(12, 6))
names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in names]
plt.bar(names, accuracies)
plt.ylabel('Accuracy')
plt.title('Classifier Comparison - Accuracy Scores')
plt.xticks(rotation=45, ha='right')
plt.ylim([0.85, 1.0])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
# plt.savefig('Excercises/exercise_02_classifier_comparison/classifier_comparison.png', dpi=150, bbox_inches='tight')
# print("Bar chart saved as 'classifier_comparison.png'")
plt.show()

# Task 3.3: Confusion matrices for best and worst classifiers
print("\n--- CONFUSION MATRICES ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

best_clf = trained_classifiers[best_clf_name]
worst_clf = trained_classifiers[worst_clf_name]

ConfusionMatrixDisplay.from_estimator(
    best_clf, X_test_scaled, y_test,
    ax=axes[0], display_labels=['Malignant', 'Benign']
)
axes[0].set_title(f'Best: {best_clf_name}')

ConfusionMatrixDisplay.from_estimator(
    worst_clf, X_test_scaled, y_test,
    ax=axes[1], display_labels=['Malignant', 'Benign']
)
axes[1].set_title(f'Worst: {worst_clf_name}')

plt.tight_layout()
# plt.savefig('Excercises/exercise_02_classifier_comparison/confusion_matrices.png', dpi=150, bbox_inches='tight')
# print("Confusion matrices saved as 'confusion_matrices.png'")
plt.show()

print("\n" + "=" * 60)
print("Part 3 Complete: Analysis and visualizations generated!")
print("=" * 60)

# =============================================================================
# PART 4: ANALYSIS QUESTIONS
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: ANALYSIS QUESTIONS")
print("=" * 60)

# Q1: Which classifier performed best? Why might that be?
print("\nQ1: Which classifier performed best? Why might that be?")
best_by_f1 = comparison_df['f1'].idxmax()
print(f"  Best: {best_by_f1} (F1={comparison_df.loc[best_by_f1, 'f1']:.4f})")
if 'SVM' in best_by_f1:
    print("  Why: RBF kernel captures complex non-linear boundaries, good for high-dimensional data")
elif 'Random Forest' in best_by_f1:
    print("  Why: Ensemble reduces overfitting, handles feature interactions well")
elif 'Logistic Regression' in best_by_f1:
    print("  Why: Good regularization, probabilistic outputs, handles linear relationships well")
else:
    print("  Why: Balanced precision/recall, good overall performance")

# Q2: Did scaling affect KNN more than Decision Tree? Why?
print("\nQ2: Did scaling affect KNN more than Decision Tree? Why?")
print("  Yes - KNN uses distance metrics, so large-scale features dominate without scaling")
print("  Decision Trees are scale-invariant - they only care about split thresholds, not distances")

# Q3: Precision vs Recall trade-offs
print("\nQ3: Which classifier for costly false positives vs false negatives?")
best_precision = comparison_df['precision'].idxmax()
best_recall = comparison_df['recall'].idxmax()
print(f"  Costly FALSE POSITIVES (need high precision): {best_precision}")
print(f"    Precision={comparison_df.loc[best_precision, 'precision']:.4f}")
print(f"  Costly FALSE NEGATIVES (need high recall): {best_recall}")
print(f"    Recall={comparison_df.loc[best_recall, 'recall']:.4f}")

# Q4: Decision Tree vs Random Forest trade-off
print("\nQ4: Trade-off between Decision Tree and Random Forest?")
dt_name = 'Decision Tree'
rf_name = 'Random Forest'
print(f"  Decision Tree: Simple/interpretable, fast, but prone to overfitting")
print(f"    Accuracy={comparison_df.loc[dt_name, 'accuracy']:.4f}")
print(f"  Random Forest: Higher accuracy, less overfitting, but less interpretable/slower")
print(f"    Accuracy={comparison_df.loc[rf_name, 'accuracy']:.4f}")

print("\n" + "=" * 60)
print("Part 4 Complete: All analysis questions answered!")
print("=" * 60)

