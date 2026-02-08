import torch
from torch import Tensor
from jaxtyping import Float

def PCA(data: Float[Tensor, "num_datapoints dim"], k: int):
    n, d = data.shape
    sample_mean = torch.mean(data, dim=0, keepdim=True)
    centered_data: Float[Tensor, "num_datapoints dim"] = data - sample_mean

    covariance: Float[Tensor, "dim dim"] = (centered_data.T @ centered_data) / n

    # --- column vectors ---
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    sorted_eigenvectors: Float[Tensor, "dim k"] = torch.flip(eigenvectors[:, -k:], dims=(1,))

    # --- centered data is row vectors so it's essenetially tranposed
    # relative to eigenvectors ---
    return centered_data @ sorted_eigenvectors