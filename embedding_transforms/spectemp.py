# Final version with finalized hyperparameters
import numpy as np
import torch
import faiss
from typing import Union
from tqdm import tqdm

def l2_normalize_rows(x: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> Union[np.ndarray, torch.Tensor]:
    """L2 normalize each row. Supports both numpy arrays and torch tensors."""
    if isinstance(x, torch.Tensor):
        x = x.float()
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norms + eps)
    else:
        x = x.astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + eps)


def calculate_snr_curve(eigenvalues: np.ndarray, tail_start_index: int = None):
    """
    Compute Local SNR (marginal signal-to-noise ratio) at different truncation dimensions k.

    Local SNR formula: SNR_local(k) = (λ_k - σ²_noise) / σ²_noise = λ_k / σ²_noise - 1

    Args:
        eigenvalues: array of eigenvalues (sorted in descending order)
        tail_start_index: start index for estimating the noise floor, defaults to the last 10%

    Returns:
        snr_curve: SNR curve
        noise_variance: estimated noise variance
    """
    if tail_start_index is None:
        tail_start_index = int(len(eigenvalues) * 0.9)  # 90% of eigenvalues; tested 0.9, 0.95, 0.85, 0.8 — results are similar, within 0.0002 error

    # Estimate noise floor — assume the tail is pure noise
    noise_variance = np.mean(eigenvalues[tail_start_index:])

    # Compute Local SNR: marginal SNR for each dimension k
    snr_list = []
    for k in range(1, tail_start_index + 1):
        lambda_k = eigenvalues[k - 1]
        local_snr = (lambda_k - noise_variance) / noise_variance
        local_snr = max(0, local_snr)  # avoid negative values
        snr_list.append(local_snr)
        
    return np.array(snr_list), noise_variance


def optimal_gamma_by_kneedle(eigenvalues: np.ndarray, target_dim: int, S: float = 0.5) -> float:
    """
    Find the optimal gamma for spectral tempering by Kneedle algorithm.

    Approach:
    - Compute the SNR curve
    - Use Kneedle algorithm to find the knee point (signal-noise boundary)
    - best_gamma = SNR(target_dim) / SNR(knee_point)

    Args:
        eigenvalues: array of eigenvalues (sorted in descending order)
        target_dim: target dimension
        S: sensitivity parameter for Kneedle algorithm; smaller values are more sensitive

    Returns:
        best_gamma: optimal gamma value
    """
    from kneed import KneeLocator

    # 1. Compute SNR curve
    snr_curve, noise_variance = calculate_snr_curve(eigenvalues)

    # 2. Find knee point using Kneedle algorithm
    target_dims = np.arange(1, len(snr_curve) + 1)
    kneedle = KneeLocator(
        target_dims,
        snr_curve,
        curve='convex',           # SNR curve bends downward
        direction='decreasing',   # SNR decreases with dimension
        S=S                       # sensitivity parameter
    )

    knee_point = kneedle.knee
    if knee_point is None:
        print("[Warning] Kneedle could not find knee point, using default knee_point = target_dim // 2")
        knee_point = max(1, target_dim // 2)

    # 3. Retrieve SNR values
    # Ensure target_dim and knee_point are within valid range
    if target_dim > len(snr_curve):
        print(f"[Warning] target_dim ({target_dim}) > len(snr_curve) ({len(snr_curve)}), using last SNR value")
        snr_target = snr_curve[-1]
    else:
        snr_target = snr_curve[target_dim - 1]

    snr_knee = snr_curve[knee_point - 1]

    # 4. Compute best_gamma
    # Avoid division by zero
    if snr_knee == 0:
        print("[Warning] SNR(knee_point) is 0, returning gamma = 1.0")
        return 1.0
    
    best_gamma = snr_target / snr_knee
    
    print(f"[Kneedle] knee_point = {knee_point}, SNR(knee) = {snr_knee:.4f}")
    print(f"[Kneedle] target_dim = {target_dim}, SNR(target) = {snr_target:.4f}")
    print(f"[Kneedle] best_gamma = SNR({target_dim}) / SNR({knee_point}) = {best_gamma:.6f}")
    
    return best_gamma


def unified_spectral_tempering_truncation(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    target_dim: int = 256,    # output dimension (e.g. 4096 -> 768)
    gamma: float = 0.05,      # whitening intensity; set to 0.02 or 0.0 to protect the top components
    auto_gamma: bool = True,  # whether to auto-compute gamma (using Kneedle algorithm)
    kneedle_S: float = 0.5,   # Kneedle sensitivity parameter (consistent with draw_singal_to_noise_ratio.py)
    epsilon: float = 1e-6,
    seed: int = 2026,
    sample_size: int = 1000000,
    batch_size: int = 100000,
    remove_mean: bool = True,
):
    """
    Spectral Tempering: dimensionality reduction and whitening transform based on SNR (signal-to-noise ratio) analysis.

    Core idea:
    1. Perform eigendecomposition of the covariance matrix to obtain principal directions and eigenvalues
    2. Compute SNR curve from eigenvalues: SNR(k) = (λ_k - σ²_noise) / σ²_noise
    3. Use Kneedle algorithm to find the knee point of the SNR curve as the signal-noise boundary
    4. Auto-compute optimal gamma: gamma = SNR(target_dim) / SNR(knee_point)
    5. Use gamma to control whitening intensity: gamma=0 preserves original scale, gamma=1 fully whitens
    6. Truncate to target_dim dimensions, discarding the noisy tail

    Transform formula: x' = (x - mean) @ U @ diag(λ^(-γ/2))

    Args:
        query_embeddings: query embeddings (n_queries, d)
        document_embeddings: document embeddings (n_docs, d)
        target_dim: output dimension
        gamma: whitening intensity in [0, 1]. 0=no whitening, 1=full whitening
        auto_gamma: whether to auto-compute gamma using Kneedle algorithm
        kneedle_S: Kneedle sensitivity; smaller values are more sensitive
        epsilon: numerical stability parameter
        seed: random seed
        sample_size: number of samples for covariance estimation
        batch_size: batch size for transform
        remove_mean: whether to subtract mean before transforming

    Returns:
        transformed_queries: transformed query embeddings
        transformed_docs: transformed document embeddings
    """

    # 0. Data preparation
    n_docs = document_embeddings.shape[0]
    n_queries = query_embeddings.shape[0]
    d = document_embeddings.shape[1]
    
    print(f"[*] Starting Unified Spectral Whitening...")
    print(f"    - Params: target_dim={target_dim}")

    # ======================================================
    # Step 1: Sample documents (operates on sampled data only)
    # ======================================================
    print(f"[*] Step 1: Sampling ({sample_size} docs) to find density basis...")
    n_total = n_docs
    real_sample_size = min(sample_size, n_total)

    np.random.seed(seed)
    indices = np.random.choice(n_total, real_sample_size, replace=False)

    # Cast sample to float64 for numerical precision in covariance and eigendecomposition
    sample_data = document_embeddings[indices].astype(np.float64)
    sample_mean = np.mean(sample_data, axis=0)  # shape: (d,)

    # ======================================================
    # Step 2: Spectral Decomposition
    # ======================================================
    print(f"[*] Step 2: Spectral Analysis on sample data...")
    # Compute covariance matrix (auto-centers)
    cov_g = np.cov(sample_data.T)
    del sample_data  # free sample data

    L, U = np.linalg.eigh(cov_g)
    del cov_g  # free covariance matrix

    # Sort in descending order
    idx = np.argsort(L)[::-1]
    L = L[idx]
    U = U[:, idx]

    # ======================================================
    # Step 2.5: Auto-compute gamma (if enabled)
    # ======================================================
    if auto_gamma:
        print(f"[*] Step 2.5: Auto-calculating gamma using Kneedle algorithm...")
        gamma = optimal_gamma_by_kneedle(L, target_dim, S=kneedle_S)
        gamma = min(gamma, 1.0)  # ensure gamma does not exceed 1.0
        print(f"    - Auto gamma: {gamma:.6f}")
    else:
        print(f"    - Using fixed gamma: {gamma}")

    # Scaling factors for the top target_dim components
    scales = (L[:target_dim] + epsilon) ** (-gamma / 2.0)

    print(f"[*] Physical Dimension Reduction: {d} -> {target_dim}")

    # Slice eigenvector matrix (keep first target_dim columns)
    U_reduced = U[:, :target_dim]  # shape: (d, target_dim)
    P = U_reduced * scales[np.newaxis, :]  # shape: (d, target_dim)
    del U, L, scales, U_reduced  # free intermediate variables

    # ======================================================
    # Step 3: Batch transform documents
    # ======================================================
    transformed_docs_list = []
    for i in tqdm(range(0, n_docs, batch_size), desc="Spectral transform docs"):
        batch = document_embeddings[i:i+batch_size].astype(np.float64)
        
        if remove_mean:
            batch_transformed = np.dot(batch - sample_mean, P)
        else:
            batch_transformed = np.dot(batch, P)
        
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_docs_list.append(batch_transformed.astype(np.float32))
        del batch, batch_transformed
    
    transformed_docs = np.vstack(transformed_docs_list)
    del transformed_docs_list

    # ======================================================
    # Step 4: Batch transform queries
    # ======================================================
    transformed_queries_list = []
    for i in tqdm(range(0, n_queries, batch_size), desc="Spectral transform queries"):
        batch = query_embeddings[i:i+batch_size].astype(np.float64)
        
        if remove_mean:
            batch_transformed = np.dot(batch - sample_mean, P)
        else:
            batch_transformed = np.dot(batch, P)
        
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_queries_list.append(batch_transformed.astype(np.float32))
        del batch, batch_transformed
    
    transformed_queries = np.vstack(transformed_queries_list)
    del transformed_queries_list, P, sample_mean

    return transformed_queries, transformed_docs