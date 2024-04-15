import json
import os
import shutil
from os import path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from modules.const import INF
from modules.logger import log


########################################################################################################################
## Attention utils
########################################################################################################################
def rescale(attention: torch.Tensor or list, mask: torch.Tensor = None):
    """Project min value to 0 and max value to 1. Assign 0 to all position if uniform attention
    
    Parameters
    ----------
    attention : torch.Tensor or list
        attention to be rescaled
    mask : torch.Tensor, optional
        mask to be applied to attention
        
    Returns
    -------
    torch.Tensor
        rescaled attention
    """
    if isinstance(attention, list):
        attention = torch.tensor(attention)
    elif isinstance(attention, np.ndarray):
        attention = torch.from_numpy(attention)
    is_instance = len(attention.shape) < 2
    if is_instance:
        attention = attention.unsqueeze(0)
    if mask is None:
        mask = torch.zeros_like(attention, dtype=torch.bool)
    v_max = torch.max(attention + mask.float() * -INF, dim=1, keepdim=True).values
    v_min = torch.min(attention + mask.float() * INF, dim=1, keepdim=True).values
    v_min[v_min == v_max] = 0.  # if v_min ==
    v_max[(v_max == 0) & (v_min == v_max)] = 1.
    rescale_attention = (attention - v_min) / (v_max - v_min)
    rescale_attention[mask] = 0.
   
    if is_instance:
        return rescale_attention.squeeze(0)
    else:
        return rescale_attention


def flatten_attention(attention, pad_mask, condition=None, normalize: str = ''):
    """Flatten attention map and filter by condition and pad_mask
    
    Parameters
    ----------
    attention : torch.Tensor
        attention map
    pad_mask : torch.Tensor
        mask of pad tokens
    condition : torch.Tensor, optional
        condition to filter attention map
    normalize : str, optional
        softmax, softmax_rescale, sigmoid
        
    Returns
    -------
    torch.Tensor
        flatten attention map
    """
    
    if condition is None:
        condition = torch.ones(attention.size(0)).type(torch.bool)
    
    if (~condition).all():
        return torch.tensor([])
    
    # Filter by condition on y
    attention = attention[condition]
    pad_mask = pad_mask[condition]
    
    # Ir normalize specify:
    if normalize == 'sigmoid':
        attention = torch.sigmoid(attention)
    if 'softmax' in normalize:
        attention = torch.softmax(attention + pad_mask.float() * -INF, dim=1)
    if 'rescale' in normalize:
        attention = rescale(attention, pad_mask)
    
    # Filter by mask
    return attention[~pad_mask]


def binarize_attention(attention: Union[np.ndarray, torch.tensor],
                       mass=None,
                       topk=None,
                       threshold=None,
                       ):
    """Get a binary vector where sum of attention value of marked tokens are more than the attention mass.

    Parameters
    ----------
    attention : array or list or `torch.Tensor`
        attention map (sum into 1)
    mass : float
        minimum attention map to retain.

    Returns
    -------
    torch.tensor or list or np.array
    """
    is_batch = True
    if len(attention.size()) == 1:
        is_batch = False
        attention = attention.unsqueeze(0)
    
    sums_row = attention.sum(dim=1)
    assert ((0.9 < sums_row) & (sums_row < 1.1)).all(), 'attention map isn\'t normalized'
    assert sum([mass is not None, topk is not None, threshold is not None]) == 1, 'Only one of `mass`, `topk`, `threshold` can be specified'
    
    DEFAUT_MASS = 0.8
    # if all parameters are None, set mass to default value
    if mass is None and topk is None and threshold is None:
        mass = DEFAUT_MASS
    
    if mass is not None:
        # binarize attention map based on total mass of attention :
        # give 1 on tokens with highest attention until total mass is reached
        
        assert 0 <= mass <= 1, 'mass must be between 0 and 1'
        
        # Sort the distribution tensor in descending order
        sorted_alpha, indices = torch.sort(attention, descending=True, dim=-1)
        # Compute the cumulative sum of the sorted tensor
        cumsum_alpha = torch.cumsum(sorted_alpha, dim=-1)
        # Find the index of the first element whose cumulative sum is >= 0.8
        index = torch.argmax((cumsum_alpha >= mass).int(), dim=-1)
        # Create a binary mask with True for the elements with cumsum_alpha >= 0.8
        b = torch.zeros_like(attention, dtype=torch.float)
        for i in range(b.shape[0]):  # Loop over the batch dimension
            b[i, indices[i, :index[i] + 1]] = 1
        
        if not is_batch:
            return b.squeeze(0)
        
        return b
    
    if threshold is not None:
        # binarize attention map based on fixed threshold
        assert 0 < threshold < 1, 'threshold must be between 0 and 1'
        b = (attention > threshold).float()
        if not is_batch:
            return b.squeeze(0)
        return b
    
    if topk is not None:
        # binarize by taking topk highest attention tokens
        assert isinstance(topk, int), 'topk must be an integer'
        assert topk > 0, 'topk must be positive'
        
        if topk > attention.size(1):
            return torch.ones_like(attention, dtype=torch.float)
        
        # Sort the distribution tensor in descending order
        sorted_alpha, indices = attention.sort(descending=True, dim=-1)
        
        b = torch.zeros_like(attention, dtype=torch.float)
        batch_indices = torch.arange(attention.size(0)).unsqueeze(1).expand(-1, topk).to(attention.device)
        topk_indices = indices[:, :topk]
        
        b[batch_indices, topk_indices] = 1.
        
        if not is_batch:
            return b.squeeze(0)
        return b
    
    raise ArithmeticError


def topk_2d(m, k, **kwargs):
    """Get topk values and indices from 2d matrix

    Parameters
    ----------
    m : torch.tensor
        2d matrix that has
    k : int
        k elements from top to get
    largest : bool
        see `torch.top.k`. Default : `True`
    **kwargs :
        arguments to be passed in `torch.topk`
    Returns
    -------

    """
    k = int(k)
    
    if k < 0 or k > m.numel():
        # if k < 0, then get all elements
        # if k is larger than number of elements in m, then get all elements
        k = m.numel()
    
    # flatten into 1d array because torch can only work on 1d
    values, top_indices = torch.topk(m.flatten(), k, **kwargs)
    
    # map back to 2d coordinate
    top_x = top_indices.div(m.shape[1], rounding_mode='trunc')
    top_y = top_indices.remainder(m.shape[1])
    
    return values, (top_x, top_y)


########################################################################################################################
## Visualization utils
########################################################################################################################

def hex2rgb(hex):
    rgb = [int(hex[i:i + 2], 16) for i in [1, 3, 5]]
    return rgb


def highlight(words: List[str], weights: Union[np.ndarray, torch.tensor, list], color: Union[str, Tuple[int]] = None, normalize_weight: bool=True):
    """Build HTML that highlights words based on its weights
    
    Parameters
    ----------
    words : list of token (str)
        1-D list iterable, containing tokens
    weights : numpy.ndarray, torch.tensor or list
        weight along text
    color : str or tuple, optional
        highlight color, in hexadecimal (ex: `#FF00FF`) or rgb (ex: `(11,12,15)`)
        
    Returns
    -------
    str
    
    Examples
    -------
    ```python
        from IPython.core.display import display, HTML
        highlighted_text = hightlight_txt(lemma1[0], a1v2)
        display(HTML(highlighted_text))
        ```
    """
    # MAX_ALPHA = 0.8
    MAX_ALPHA = 1.0
    
    # convert all type to torch tensor
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    elif isinstance(weights, list):
        weights = torch.tensor(weights)
    
    weights = weights.float()
    
    # normalize weights
    if normalize_weight:
        w_min, w_max = torch.min(weights), torch.max(weights)
        w_norm = (weights - w_min) / ((w_max - w_min) + (w_max == w_min) * w_max)
    else:
        w_norm = weights
    
    # make color
    # change to rgb if given color is hex
    if color is None:
        color = [135, 206, 250]
    elif color[0] == '#' and len(color) == 7:
        color = hex2rgb(color)
    w_norm = (w_norm * MAX_ALPHA).tolist()
    
    # wrap each token in a span
    highlighted_words = []
    for word, w in zip(words, w_norm):
        if w > 0:
            span = f'<span style="background-color:rgba{(*color, w)};">' + word + ' </span>'
        else:
            span = word + ' '
        highlighted_words.append(span)
    #highlighted_words = [f'<span style="background-color:rgba{(*color, w)};">' + word + '</span>' for word, w in zip(words, w_norm)]
    
    # concatenate spans into a string
    return ''.join(highlighted_words)


########################################################################################################################
## Lightining utils
########################################################################################################################

def report_score(scores: dict, logger, score_dir=None) -> None:
    """Report scores into score.json and logger
    
    Parameters
    ----------
    scores : dict
        dictionary that has reported scores
    logger : Tensorboard logger
        Report into hyperparameters
    score_dir : str, optional
        directory to find score.json
    
    Returns
    -------
    None
    """
    
    # remove 'TEST/' from score dicts:
    scores = [{k.replace('TEST/', ''): v for k, v in s.items()} for s in scores]
    
    for idx, score in enumerate(scores):
        log.info(score)
        logger.log_metrics(score)
        
        if score_dir is not None:
            os.makedirs(score_dir, exist_ok=True)
            src = path.join(logger.log_dir, 'hparams.yaml')
            dst = path.join(score_dir, 'hparams.yaml')
            shutil.copy2(src, dst)
        
        score_path = path.join(score_dir or logger.log_dir, f'score{"" if idx == 0 else "_" + str(idx)}.json')
        
        with open(score_path, 'w') as fp:
            json.dump(score, fp, indent='\t')
            log.info(f'Score is saved at {score_path}')


def get_num_workers() -> int:
    """Get maximum logical workers that a machine has

    Returns
    -------
    num_workers : int
        maximum workers number possible
    """
    if hasattr(os, 'sched_getaffinity'):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass
    
    num_workers = os.cpu_count()
    return num_workers if num_workers is not None else 0

########################################################################################################################
## Generic Python utils
########################################################################################################################


def flatten_dict(nested: dict, sep: str = '.') -> dict:
    """
    Convert a nested dictionary into a flatt dictionary
    
    Parameters
    ----------
    nested : dict
        nested dictionary to be flattened
    sep : str
        separator between parent key and child key

    Returns
    -------
    dict
        flat dictionary

    """
    assert isinstance(nested, dict), f"Only flatten dictionary, nested given is of type {type(nested)}"
    
    flat = dict()
    
    for current_key, current_value in nested.items():
        
        # if value is a dictionary, then concatenate its key with current key
        if isinstance(current_value, dict):
            flat_item = flatten_dict(current_value, sep=sep)
            
            flat.extend()
            
            for child_key, child_value in flat_item.items():
                flat[current_key + sep + child_key] = child_value
        
        else:
            flat[current_key] = current_value
    
    return flat


def quick_flatten_dict(nested: dict, sep: str = '.') -> dict:
    """New version of how to flat a dictionary using pandas
    
    Parameters
    ----------
    nested : dict
        nested dictionary to be flattened
    sep : str
        separator between parent key and child key

    Returns
    -------
    dict
        flat dictionary

    """
    return pd.json_normalize(nested, sep=sep).to_dict(orient='records')[0]


def map_list2dict(batch: Union[List[Dict], Dict]) -> dict:
    """convert list of dict to dict of list
    
    Parameters
    ----------
    batch : List[Dict] or Dict
        batch of dictionaries
        
    Returns
    -------
    dict
        dictionary where data are batched in each key.
    """
    if isinstance(batch, dict):
        return {k: list(v) for k, v in batch.items()}  # handle case where no batch
    return {k: [row[k] for row in batch] for k in batch[0]}


def recursive_dict2list( value, i):
    """Unpack a value from a nested structure at the given index."""
    if isinstance(value, list):
        return value[i]
    elif isinstance(value, dict):
        return {k: recursive_dict2list(v, i) for k, v in value.items()}
    else:
        return value


def map_dict2list(batch):
    batch_size = len(next(iter(batch.values())))
    result = []
    
    for i in range(batch_size):
        item = {key: recursive_dict2list(value, i) for key, value in batch.items()}
        result.append(item)
    
    return result


def recursive_list2dict(batch: Union[List[Dict], Dict]):
    """Convert list of dict to dict of list recursively"""
    
    if isinstance(batch, list):
        
        if isinstance(batch[0], dict):
            batch = {k: [row[k] for row in batch] for k in batch[0]}
        
        elif isinstance(batch[0], list):
            batch = [item for sub_list in batch for item in sub_list]
    
    if isinstance(batch, dict):
        
        for k in batch:
            # flatten all list of dict
            batch[k] = recursive_list2dict(batch[k])
    
    return batch


def recursive_tensor2list(batch: [Dict]):
    """Convert tensor to list recursively"""
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.tolist()
        elif isinstance(v, dict):
            batch[k] = recursive_tensor2list(v)
    return batch

def map_np2list(df: pd.DataFrame):
    """Auto convert numpy columns into list columns
    
    Parameters
    ----------
    df : pd.DataFrame
        entire data

    Returns
    -------
    df : pd.DataFrame
        formatted data
    """
    
    return df.apply(lambda column: [c.tolist() for c in column] if isinstance(column[0], np.ndarray) else column)



