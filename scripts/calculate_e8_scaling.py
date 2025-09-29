# filename: calculate_e8_scaling.py

import numpy as np
from scipy.spatial import cKDTree
import itertools
import time
import csv

print("--- E8 Scaling Calculation Script ---")

# ===================================================================
# PART 1: E8 SHELL GENERATOR
# ===================================================================
# This function is the slow, one-time setup part of the script.
def generate_e8_shell(n):
    if not hasattr(generate_e8_shell, "shells_by_n"):
        print("Generating E8 point cloud for shell lookup... (This may take several minutes)")
        # 4 iterations is required to generate shells up to n~16
        iterations = 6
        points = {tuple(np.zeros(8))}
        current_layer = {tuple(np.zeros(8))}
        
        # Define the 240 root vectors of E8
        roots = []
        for i, j in itertools.combinations(range(8), 2):
            for si, sj in itertools.product([-1, 1], repeat=2):
                vec = np.zeros(8); vec[i], vec[j] = si, sj
                roots.append(tuple(vec))
        for signs in itertools.product([-0.5, 0.5], repeat=8):
            if np.sum(np.array(signs) < 0) % 2 == 0:
                roots.append(signs)
        
        # Iteratively build the point cloud
        for iter_num in range(iterations):
            print(f"  ...Pre-computation iteration {iter_num+1}/{iterations}, current cloud size: {len(points)}")
            next_layer = set()
            for p_tuple in current_layer:
                p = np.array(p_tuple)
                for r_tuple in roots:
                    r = np.array(r_tuple)
                    next_p = tuple(p + r)
                    if next_p not in points:
                        next_layer.add(next_p)
            points.update(next_layer)
            current_layer = next_layer

        point_cloud = np.array(list(points))
        norms_sq = np.sum(point_cloud**2, axis=1)
        
        generate_e8_shell.shells_by_n = {}
        for i in range(len(point_cloud)):
            norm_sq = norms_sq[i]
            if abs(norm_sq % 2) < 1e-9:
                shell_n = int(round(norm_sq / 2))
                if shell_n == 0: continue
                if shell_n not in generate_e8_shell.shells_by_n:
                    generate_e8_shell.shells_by_n[shell_n] = []
                generate_e8_shell.shells_by_n[shell_n].append(point_cloud[i])
        for shell_n in generate_e8_shell.shells_by_n:
             generate_e8_shell.shells_by_n[shell_n] = np.array(generate_e8_shell.shells_by_n[shell_n])
        print("E8 point cloud generation complete.")

    if n in generate_e8_shell.shells_by_n:
        return generate_e8_shell.shells_by_n[n]
    else:
        raise ValueError(f"Shell n={n} not pre-generated. Increase pre-computation iterations.")

# ===================================================================
# PART 2: PROJECTION AND EXPERIMENT FUNCTIONS
# ===================================================================
def random_projection_8_to_3(seed=0):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((8, 3))
    Q, _ = np.linalg.qr(G)
    return Q

#def min_spacing(points3):
#    if len(points3) < 2: return np.inf
#    tree = cKDTree(points3)
#    dists, _ = tree.query(points3, k=2)
#    return dists[:, 1].min()

# ===================================================================

def spacing_stat(points3, k=1, q=0.5):
    """
    Calculates a robust spacing statistic.
    k: k-th nearest neighbor (k=1 for the closest neighbor)
    q: quantile (q=0.5 for the median)
    """
    if len(points3) < k+1:
        return np.inf
    tree = cKDTree(points3)
    # Query for k+1 neighbors because the first (k=0) is the point itself
    dists, _ = tree.query(points3, k=k+1)
    # Get the distances to the k-th nearest neighbor (index k)
    dk = dists[:, k]
    # Return the q-th quantile of these distances
    return np.quantile(dk, q)

def run_experiment_sparse(n_values, projection_matrix):
    # This list will store the typical spacing for each n
    ell_typical = []
    print("Pre-loading all required shells...")
    all_shells = {}
    try:
        for n in range(1, max(n_values) + 1):
            all_shells[n] = generate_e8_shell(n)
    except ValueError as e:
        print(f"ERROR: {e}")
        return [], []
    print("Shells loaded successfully.")

    # --- CHANGE #1 ---
    # Calculate the REFERENCE spacing at n=1 using the new robust statistic.
    # We also rename the variable for clarity.
    s_stat_1 = spacing_stat(all_shells[1] @ projection_matrix, k=1, q=0.5)
    
    if s_stat_1 == np.inf:
        print("ERROR: Could not calculate a valid reference spacing for n=1.")
        return [], []

    for n_target in n_values:
        start_time = time.time()
        S = np.vstack([all_shells[i] for i in range(1, n_target + 1)])
        X = S @ projection_matrix
        
        # --- CHANGE #2 ---
        # Calculate the TYPICAL spacing at step n using the new robust statistic.
        s_stat_n = spacing_stat(X, k=1, q=0.5)
        ell_typical.append(s_stat_n)
        
        duration = time.time() - start_time
        
        # --- CHANGE #3 ---
        # Update the print statement to be more accurate.
        print(f"Processed up to n={n_target} ({len(S)} points). Median spacing: {s_stat_n:.6f}. Time: {duration:.2f}s")
        
    ell_typical = np.array(ell_typical)
    
    # --- CHANGE #4 ---
    # Calculate the scale factor using the consistent robust spacing metric.
    a = s_stat_1 / ell_typical if len(ell_typical) > 0 else np.array([])
    return ell_typical, a

# ===================================================================
# PART 3: MAIN EXECUTION BLOCK
# ===================================================================
if __name__ == "__main__":
    # Define the sparse set of n values to test
    #n_points_to_test = np.unique(np.logspace(np.log10(2), np.log10(12), 12).astype(int))
    #n_points_to_test = np.unique(np.logspace(np.log10(2), np.log10(19), 19).astype(int))
    n_points_to_test = np.unique(np.logspace(np.log10(2), np.log10(28), 25).astype(int))
    print(f"Will test sparse n values: {n_points_to_test}")

    # Create the projection matrix
    PI = random_projection_8_to_3(seed=42) # Use a fixed seed for reproducibility

    # Run the main experiment
    _, scale_factor_a = run_experiment_sparse(n_values=n_points_to_test, projection_matrix=PI)

    # Save the results to a CSV file
    if len(scale_factor_a) > 0:
        output_filename = "data/e8_scaling_data.csv"
        print(f"\nSaving results to {output_filename}...")
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(['n', 'scale_factor_a'])
            # Write the data rows
            for n_val, a_val in zip(n_points_to_test, scale_factor_a):
                writer.writerow([n_val, a_val])
        print("Save complete.")
    else:
        print("\nExperiment failed to produce data. Nothing to save.")

    print("\n--- Script Finished ---")
