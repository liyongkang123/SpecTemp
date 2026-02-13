import numpy as np
import torch
from typing import Union
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# ======================================================
# Baseline: Matryoshka Truncation
# ======================================================
def matryoshka_truncation(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    target_dim: int = 512
):
    """
    Matryoshka direct truncation of the first k dimensions
    """
    compressed_queries = query_embeddings[:, :target_dim].copy()
    compressed_docs = document_embeddings[:, :target_dim].copy()

    # L2 normalization
    compressed_queries = l2_normalize_rows(compressed_queries)
    compressed_docs = l2_normalize_rows(compressed_docs)
    
    return compressed_queries, compressed_docs

# ======================================================
# Baseline: Random Truncation
# ======================================================
def random_truncation(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    target_dim: int = 512,
    seed: int = 2026,
):
    """
    Random selection of k dimensions
    """
    np.random.seed(seed)
    random_indices = np.random.choice(query_embeddings.shape[1], target_dim, replace=False)
    compressed_queries = query_embeddings[:, random_indices].copy()
    compressed_docs = document_embeddings[:, random_indices].copy()

    # L2 normalization
    compressed_queries = l2_normalize_rows(compressed_queries)
    compressed_docs = l2_normalize_rows(compressed_docs)
    
    return compressed_queries, compressed_docs


# ======================================================
# Baseline: PCA Truncation
# ======================================================
def pca_truncation(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    target_dim: int = 512,
    sample_size: int = 1000000,
    batch_size: int = 100000,
    seed: int = 2026,
):
    """
    PCA dimensionality reduction (numpy version)
    Fit on a sample of documents, then transform queries and documents.

    PCA transform formula: x' = (x - mean) @ U[:, :k]
    """
    n_docs = document_embeddings.shape[0]
    n_queries = query_embeddings.shape[0]
    real_sample_size = min(sample_size, n_docs)
    print(f"[PCA] Using sample_size={real_sample_size} for fitting")

    # 1. Sample to compute mean and covariance (float64 for numerical precision)
    np.random.seed(seed)
    sample_indices = np.random.choice(n_docs, real_sample_size, replace=False)
    sample_data = document_embeddings[sample_indices].astype(np.float64)

    # Compute mean
    sample_mean = np.mean(sample_data, axis=0)  # (n_dim,)

    # Compute covariance matrix (np.cov auto-centers)
    cov = np.cov(sample_data.T)
    del sample_data

    # 2. Eigenvalue decomposition
    L, U = np.linalg.eigh(cov)
    del cov

    # Sort in descending order
    idx = np.argsort(L)[::-1]
    U = U[:, idx]

    # Take first target_dim principal components (PCA projects without whitening)
    U_reduced = U[:, :target_dim]  # (n_dim, target_dim)
    del U

    print(f"[PCA] Physical Dimension Reduction: {document_embeddings.shape[1]} -> {target_dim}")

    # 3. Batch transform documents
    transformed_docs_list = []
    for i in tqdm(range(0, n_docs, batch_size), desc="PCA transform docs"):
        batch = document_embeddings[i:i+batch_size].astype(np.float64)
        batch_centered = batch - sample_mean
        batch_transformed = batch_centered @ U_reduced
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_docs_list.append(batch_transformed.astype(np.float32))
        del batch, batch_centered, batch_transformed

    transformed_document_embeddings = np.vstack(transformed_docs_list)
    del transformed_docs_list

    # 4. Batch transform queries
    transformed_queries_list = []
    for i in tqdm(range(0, n_queries, batch_size), desc="PCA transform queries"):
        batch = query_embeddings[i:i+batch_size].astype(np.float64)
        batch_centered = batch - sample_mean
        batch_transformed = batch_centered @ U_reduced
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_queries_list.append(batch_transformed.astype(np.float32))
        del batch, batch_centered, batch_transformed

    transformed_query_embeddings = np.vstack(transformed_queries_list)
    del transformed_queries_list, sample_mean, U_reduced
    
    return transformed_query_embeddings, transformed_document_embeddings


# ======================================================
# Baseline: Whitening-k Transform
# ======================================================
def whitening_k_truncation(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    target_dim: int = 512,
    sample_size: int = 1000000,
    batch_size: int = 100000,
    seed: int = 2026,
    beta: float = 1.0,
    gamma: float = 1.0
):
    """
    Whitening-k transform (numpy version)
    Compute covariance from samples, batch transform to save memory.

    Args:
        beta: mean scaling factor, controls degree of mean subtraction. 1.0=full subtraction, 0.0=none
        gamma: whitening intensity, 1.0=full whitening
    """
    n_docs, n_dim = document_embeddings.shape
    n_queries = query_embeddings.shape[0]
    real_sample_size = min(sample_size, n_docs)

    # 1. Sample to compute mean and covariance (float64 for numerical precision)
    np.random.seed(seed)
    sample_indices = np.random.choice(n_docs, real_sample_size, replace=False)
    sample_data = document_embeddings[sample_indices].astype(np.float64)

    # Compute mean (scaled by beta to control degree of mean subtraction)
    sample_mean = np.mean(sample_data, axis=0) * beta  # (n_dim,)

    # Compute covariance matrix (np.cov auto-centers)
    cov = np.cov(sample_data.T)
    del sample_data

    # 2. Eigenvalue decomposition
    L, U = np.linalg.eigh(cov)
    del cov

    # Sort in descending order
    idx = np.argsort(L)[::-1]
    L = L[idx]
    U = U[:, idx]

    # Take first target_dim principal components and build whitening kernel
    L = L[:target_dim]
    U = U[:, :target_dim]
    eps = 1e-6
    kernel = U * ((L + eps) ** (-(gamma / 2.0)))  # (n_dim, target_dim)
    del L, U

    # 3. Batch transform documents
    transformed_docs_list = []
    for i in tqdm(range(0, n_docs, batch_size), desc="Whitening transform docs"):
        batch = document_embeddings[i:i+batch_size].astype(np.float64)
        batch_centered = batch - sample_mean
        batch_transformed = batch_centered @ kernel
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_docs_list.append(batch_transformed.astype(np.float32))
        del batch, batch_centered, batch_transformed

    transformed_document_embeddings = np.vstack(transformed_docs_list)
    del transformed_docs_list

    # 4. Batch transform queries
    transformed_queries_list = []
    for i in tqdm(range(0, n_queries, batch_size), desc="Whitening transform queries"):
        batch = query_embeddings[i:i+batch_size].astype(np.float64)
        batch_centered = batch - sample_mean
        batch_transformed = batch_centered @ kernel
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_queries_list.append(batch_transformed.astype(np.float32))
        del batch, batch_centered, batch_transformed

    transformed_query_embeddings = np.vstack(transformed_queries_list)
    del transformed_queries_list, sample_mean, kernel
    
    return transformed_query_embeddings, transformed_document_embeddings

# ======================================================
# Baseline: Random Projection
# ======================================================

def random_projection_truncation(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    target_dim: int = 512,
    batch_size: int = 100000,
    seed: int = 2026,
):
    """
    Random Projection dimensionality reduction (numpy version)
    Uses a Gaussian random projection matrix to project high-dimensional embeddings to a lower-dimensional space.
    According to the Johnson-Lindenstrauss lemma, random projection approximately preserves pairwise distances.
    """
    n_docs = document_embeddings.shape[0]
    n_queries = query_embeddings.shape[0]
    original_dim = document_embeddings.shape[1]

    # Generate Gaussian random projection matrix R: (original_dim, target_dim)
    # Scale by 1/sqrt(target_dim) to preserve distances
    np.random.seed(seed)
    R = np.random.randn(original_dim, target_dim).astype(np.float64) / np.sqrt(target_dim)

    # Batch transform documents
    transformed_docs_list = []
    for i in tqdm(range(0, n_docs, batch_size), desc="Random projection docs"):
        batch = document_embeddings[i:i+batch_size].astype(np.float64)
        batch_transformed = batch @ R
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_docs_list.append(batch_transformed.astype(np.float32))

    transformed_document_embeddings = np.vstack(transformed_docs_list)
    del transformed_docs_list

    # Batch transform queries
    transformed_queries_list = []
    for i in tqdm(range(0, n_queries, batch_size), desc="Random projection queries"):
        batch = query_embeddings[i:i+batch_size].astype(np.float64)
        batch_transformed = batch @ R
        batch_transformed = l2_normalize_rows(batch_transformed)
        transformed_queries_list.append(batch_transformed.astype(np.float32))
    
    transformed_query_embeddings = np.vstack(transformed_queries_list)
    del transformed_queries_list, R
    
    return transformed_query_embeddings, transformed_document_embeddings