import torch
from torch import Tensor
from jaxtyping import Float


def cosine_similarity(
    x: Float[Tensor, "N C"] | Float[Tensor, "B N C"], similarity_index: int
) -> Float[Tensor, "N"] | Float[Tensor, "B N"]:
    """
    Args:
        x: input sequence of vectors to compare (can be batched)
        similatiry_index: the index of the element you would like to stem comparison from
    Returns:
        A tensor of float values between [-1, 1] represneting the cosine similarity scores


    To get the cosine angle between two vectors, we use the dot product.
    Namely, two vectors a and b have the dot product
    a . b = ||a||*||b||cos(theta)
    The values for cos(theta) will of course range from [-1, 1]
    """
    if x.ndim == 2:
        x.unsqueeze_(0)
    B, N, C = x.shape
    assert (
        similarity_index >= 0 and similarity_index < N
    ), f"index out of bounds for {N} comparison vectors"

    comparison_vector: Float[Tensor, "B C 1"] = x[:, similarity_index, None].transpose(
        1, 2
    )

    similarity_scores: Float[Tensor, "B N"] = (x @ comparison_vector) / (
        (torch.linalg.norm(comparison_vector) * torch.linalg.norm(x, axis=-1)).unsqueeze_(-1)
    )

    return (
        similarity_scores
        if similarity_scores.shape[0] != 1
        else similarity_scores.squeeze_(0)
    )
