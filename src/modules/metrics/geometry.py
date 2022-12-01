# metrics for the evaluation of the geometry
import torch
from torch import Tensor
from typing import Optional, Tuple


def cosine_sim(W: Tensor, padding_mask: Tensor, normalize: str = "mean") -> Tensor:
    """ Cosine similarity function

    The objective of this function is to calculate the

    Some notations :

    N : the batch size
    T(s) : the number of tokens in the treated sentence
    d : the embedding dimension

    Args:
        W: Tensor. tensor of the shape (N, T(s), d)
        padding_mask: Tensor<bool>. Boolean tensor of shape (N, T(s)) == 1 iff the token is a padding one
        normalize: str. Indicate the shape of the output. If normalize is mean we proceed a mean over a batch
                        else, we return the tensor of the shape [N]

    Returns:
        float in the range [-1, 1] this float describes the geometry of
    """

    assert len(W.shape) == 3, "error : must give a batch tensor of shape [N, T(s), d]"
    assert len(padding_mask.shape) == 2, "error : must give a batch tensor of shape [N, T(s)]"

    N, T, d = W.shape

    # >> dot product between embeddings
    dot_prod = torch.bmm(W, torch.transpose(W, 1, 2))
    v = torch.diagonal(dot_prod, offset=0, dim1=1, dim2=2).unsqueeze(-1)
    nms = torch.sqrt(torch.bmm(v, torch.transpose(v, 1, 2)))
    s_mat = dot_prod / (nms + 1e-16)

    # >> get rid of the padding tokens
    buff = (~padding_mask).float().unsqueeze(1).repeat(1, T, 1)
    s_mat = s_mat * buff
    s_mat = s_mat * torch.transpose(buff, 1, 2)

    # >> for each sentences s : get the quantity T(s)
    t_s = (~padding_mask).float().sum(dim=-1)

    # >> proceed the metric calculus
    den = t_s * (t_s - 1)
    num = s_mat.sum(dim=-1).sum(dim=-1) - torch.diagonal(s_mat, offset=0, dim1=1, dim2=2).sum(dim=-1)

    if normalize == "mean":
        # we normalize the quantity over the batch
        res = (num / den).mean()
    else:
        res = num / den

    return res


def effective_rank(W: Tensor):
    """ Effective rank calculus

    Args:
        W: Tensor. An embedding matrix

    Returns:
        Return the effective rank of the matrix
    """
    assert W.shape[0] >= W.shape[1], "error : the erank metrics can't be computed ! not enough data"
    s = torch.linalg.svdvals(W)
    sum_s = s.sum()
    return torch.exp(-((s / sum_s) * torch.log(s / sum_s)).sum())
