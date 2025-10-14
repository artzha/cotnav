import numpy as np
from scipy.spatial.distance import cdist


def compute_accuracy(arcs: np.ndarray, probs: np.ndarray, odom: np.ndarray) -> float:
    """
    arcs: (B, K, N, 3), probs: (B, K), odom: (B, M, 3)
    Returns average accuracy over batch.
    """
    B, K, N, _ = arcs.shape
    
    # Get best prediction for each batch element (B,)
    selected_k_indices = np.argmax(probs, axis=1)
    
    # Compute Hausdorff distances for all combinations (B, K)
    hausdorff_distances = np.zeros((B, K))
    for b in range(B):
        for k in range(K):
            hausdorff_distances[b, k] = hausdorff_xyz(arcs[b, k], odom[b])
    
    # Find ground truth indices (best Hausdorff distance for each batch) (B,)
    ground_truth_indices = np.argmin(hausdorff_distances, axis=1)
    
    # Compute accuracy as fraction of correct predictions
    acc = np.mean(selected_k_indices == ground_truth_indices)
    
    return acc

def compute_hausdorff_distance(arcs: np.ndarray, probs: np.ndarray, odom: np.ndarray, oneway=True) -> float:
    """
    arcs: (B, K, N, 3), probs: (B, K), odom: (B, M, 3)
    Returns average hdist(model, odom) over batch.
    """
    B, K, N, _ = arcs.shape
    total_distance = 0.0
    selected_k_indices = np.argmax(probs, axis=1)
    for b in range(B):
        total_distance += hausdorff_xyz(arcs[b, selected_k_indices[b]], odom[b], oneway=oneway)
    return total_distance / B

def compute_relative_hausdorff_distance(arcs: np.ndarray, probs: np.ndarray, odom: np.ndarray, oneway=True) -> float:
    """
    arcs: (B, K, N, 3), probs: (B, K), odom: (B, M, 3)
    Returns average hdist(model, odom) - hdist(gt, odom) over batch.
    """
    B, K, N, _ = arcs.shape
    total_distance = 0.0
    selected_k_indices = np.argmax(probs, axis=1)
    for b in range(B):
        odom_distance = min([hausdorff_xyz(arcs[b, k], odom[b]) for k in range(K)])
        total_distance += hausdorff_xyz(arcs[b, selected_k_indices[b]], odom[b], oneway=oneway) - odom_distance
    return total_distance / B

def hausdorff_xyz(A: np.ndarray, B: np.ndarray, oneway=True) -> float:
    """Oneway Hausdorff distance between two polylines A(N,3) and B(M,3)."""
    if A.size == 0 or B.size == 0: 
        return np.inf
    D = cdist(A, B)  # (N,M)
    if oneway:
        return float(D.min(axis=1).max())
    else:
        return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


if __name__ == "__main__":
    print("Testing compute_accuracy with specific probabilities...")
    
    # Create very specific test case where we can manually verify accuracy
    B, K, N, M = 3, 4, 5, 5  # 3 batches, 4 candidates each, 5 points each
    
    # Create specific trajectories
    arcs = np.zeros((B, K, N, 3))
    
    # Batch 0: Create 4 different trajectories
    arcs[0, 0] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])  # Straight line x
    arcs[0, 1] = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]])  # Straight line y  
    arcs[0, 2] = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0]])  # Diagonal
    arcs[0, 3] = np.array([[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14]])  # Far away
    
    # Batch 1: Similar but different
    arcs[1, 0] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]])  # Straight line z
    arcs[1, 1] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])  # Straight line x
    arcs[1, 2] = np.array([[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]])  # Different diagonal
    arcs[1, 3] = np.array([[20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24]])  # Very far
    
    # Batch 2: Another set
    arcs[2, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]])  # Straight line y
    arcs[2, 1] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]])  # Straight line z
    arcs[2, 2] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])  # Straight line x
    arcs[2, 3] = np.array([[30, 30, 30], [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34]])  # Very far
    
    # Create odometry that matches specific arcs
    odom = np.zeros((B, M, 3))
    odom[0] = arcs[0, 1]  # Batch 0: odom matches candidate 1 (y-line)
    odom[1] = arcs[1, 0]  # Batch 1: odom matches candidate 0 (z-line)  
    odom[2] = arcs[2, 2]  # Batch 2: odom matches candidate 2 (x-line)
    
    # Create specific probabilities
    probs = np.array([
        [0.1, 0.7, 0.1, 0.1],  # Batch 0: predicts candidate 1 (CORRECT - matches odom)
        [0.2, 0.6, 0.1, 0.1],  # Batch 1: predicts candidate 1 (WRONG - should be 0)
        [0.1, 0.1, 0.8, 0.0],  # Batch 2: predicts candidate 2 (CORRECT - matches odom)
    ])
    
    print("Manual verification:")
    print("Batch 0: Prediction=1, Ground truth=1 → CORRECT")
    print("Batch 1: Prediction=1, Ground truth=0 → WRONG") 
    print("Batch 2: Prediction=2, Ground truth=2 → CORRECT")
    print("Expected accuracy: 2/3 = 0.6667")
    print()
    
    # Show the setup
    print("Test setup:")
    print(f"arcs shape: {arcs.shape}")
    print(f"probs shape: {probs.shape}")
    print(f"odom shape: {odom.shape}")
    print(f"probs = \n{probs}")
    print()
    
    # Manual calculation
    selected_k = np.argmax(probs, axis=1)
    print(f"Selected candidates (argmax of probs): {selected_k}")
    
    # Calculate Hausdorff distances manually for verification
    print("\nHausdorff distances for each batch:")
    for b in range(B):
        print(f"Batch {b}:")
        distances = []
        for k in range(K):
            dist = hausdorff_xyz(arcs[b, k], odom[b])
            distances.append(dist)
            print(f"  Candidate {k}: {dist:.4f}")
        gt_idx = np.argmin(distances)
        print(f"  Ground truth candidate: {gt_idx}")
        print(f"  Predicted candidate: {selected_k[b]}")
        print(f"  Correct: {gt_idx == selected_k[b]}")
        print()
    
    # Test the function
    accuracy = compute_accuracy(arcs, probs, odom)
    print(f"Computed accuracy: {accuracy:.4f}")
    print(f"Expected accuracy: 0.6667")
    print(f"Match: {abs(accuracy - 2/3) < 0.001}")