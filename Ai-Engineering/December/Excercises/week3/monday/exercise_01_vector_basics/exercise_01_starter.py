"""
Exercise 01: Vector Basics - Starter Code

This is a conceptual exercise primarily done by hand, but this file
provides Python verification for your hand calculations.

Run this AFTER completing the pen-and-paper calculations to verify.
"""

import numpy as np

# ============================================================================
# VERIFICATION CODE - Check your hand calculations
# ============================================================================

print("=" * 60)
print("Exercise 01: Vector Basics - Verification")
print("=" * 60)

# Part 1: Euclidean Distance
print("\n--- PART 1: VERIFY YOUR EUCLIDEAN DISTANCES ---")

# Task 1.1: A = [2, 3], B = [5, 7]
A = np.array([2, 3])
B = np.array([5, 7])

# TODO: Calculate by hand first, then uncomment to verify
# your_answer_1_1 = ???
expected_1_1 = np.linalg.norm(A - B)
print(f"Task 1.1: d(A, B) = {expected_1_1:.2f}")
# print(f"Your answer: {your_answer_1_1}, Correct: {np.isclose(your_answer_1_1, expected_1_1)}")


# Task 1.2: P = [1, 2, 3], Q = [4, 0, 3]
P = np.array([1, 2, 3])
Q = np.array([4, 0, 3])

# TODO: Calculate by hand first, then verify
# your_answer_1_2 = ???
expected_1_2 = np.linalg.norm(P - Q)
print(f"Task 1.2: d(P, Q) = {expected_1_2:.2f}")


# Part 2: Cosine Similarity
print("\n--- PART 2: VERIFY YOUR COSINE CALCULATIONS ---")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


# Task 2.1: Dot product of A = [1, 2] and B = [3, 4]
A2 = np.array([1, 2])
B2 = np.array([3, 4])

# TODO: Calculate by hand first
# your_dot_product = ???
expected_dot = np.dot(A2, B2)
print(f"Task 2.1: A · B = {expected_dot}")


# Task 2.2: Magnitude of A = [1, 2]
# your_magnitude = ???
expected_magnitude = np.linalg.norm(A2)
print(f"Task 2.2: ||A|| = {expected_magnitude:.4f}")


# Task 2.3: Cosine similarity
# your_cosine = ???
expected_cosine = cosine_similarity(A2, B2)
print(f"Task 2.3: cos(θ) = {expected_cosine:.4f}")


# Task 2.4: Special cases
print("\n--- TASK 2.4: SPECIAL CASES ---")

# Same direction
same_dir_a = np.array([3, 0])
same_dir_b = np.array([5, 0])
print(f"Same direction [3,0]·[5,0]: {cosine_similarity(same_dir_a, same_dir_b):.1f}")

# Perpendicular  
perp_a = np.array([1, 0])
perp_b = np.array([0, 1])
print(f"Perpendicular [1,0]·[0,1]: {cosine_similarity(perp_a, perp_b):.1f}")

# Parallel different lengths
parallel_a = np.array([1, 1])
parallel_b = np.array([2, 2])
print(f"Parallel [1,1]·[2,2]: {cosine_similarity(parallel_a, parallel_b):.1f}")


# Part 3: Comparison Table
print("\n--- PART 3: COMPARISON TABLE ---")
print("""
| Property              | Euclidean Distance | Cosine Similarity   |
|-----------------------|-------------------|---------------------|
| Range                 | 0 to ∞            | -1 to 1             |
| "Similar" means       | Low (near 0)      | High (near 1)       |
| Sensitive to magnitude| Yes               | No                  |
""")

print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)
