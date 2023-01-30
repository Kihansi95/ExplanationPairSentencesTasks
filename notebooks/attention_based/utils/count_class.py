import torch
from torch import Tensor
import pandas as pd
import numpy as np
from pandas import DataFrame


def compute_cos(rep: Tensor, vectors: Tensor) -> Tensor:
    """compute_cos

    This function will compute the cosines between a vector and a 2D tensor.
    Rep for the interpretation will be the representative of a certain class.

    Args:
        rep (Tensor): the main vector we want to make the comparison with
        vectors (Tensor): the vectors we want to now in which class they are.

    Returns:
        a 1D tensor v where v_i = cos(rep, vectors[i, :])
    """
    assert len(rep.shape) == 1, "error : batch is given"
    assert rep.shape[0] == vectors.shape[1], "errors : got unexpected dimension error"

    dot_prod = torch.matmul(vectors, rep)
    nms = torch.norm(vectors, dim=-1)

    # we return the cosine similarity
    return dot_prod / (nms * torch.norm(rep))


def search_rep(model, dm, mean_calc: bool = False):
    """

    Args:
        model: the model
        dm: the data module (data for the model)

    Returns:

    """
    DEVICE = model.device  # to have no device conflict.

    res = {
        f"class_{i}": None for i in range(dm.num_class)
    }

    for class_rep in range(dm.num_class):
        class_rep_found = False  # indicate if we found the representative of class_rep
        # search for a rep of the class_rep class
        for id_batch, batch in enumerate(dm.test_dataloader()):

            # batch elements
            y_true = batch["y_true"].to(DEVICE)
            bs = y_true.shape[0]  # the batch size

            ids = batch["token_ids"].to(DEVICE)
            padding_mask = batch["padding_mask"].bool().to(DEVICE)

            # output of the model
            output = model(ids=ids, mask=padding_mask)

            # the prediction we make
            cl = output["logits"].argmax(dim=-1)

            mask = (~padding_mask).float().unsqueeze(-1).repeat(1, 1, 300)
            nb_tokens = (~padding_mask).sum(dim=-1).unsqueeze(-1).repeat(1, 300)

            for i in range(bs):
                # prediction equals the class and the prediction is good
                # we treat here the element i of the batch.
                if cl[i] == class_rep and cl[i] == y_true[i]:
                    # we found a rep of the class class_rep

                    # initiat the lists
                    rep_k = []
                    rep_v = []
                    rep_emb = []

                    for lay in range(len(output["value_embeddings"])):
                        if ~mean_calc:
                            rep_k.append(output["key_embeddings"][lay][i, 0, :])
                            rep_v.append(output["value_embeddings"][lay][i, 0, :])
                        else:
                            # we delete all the masks embeddings
                            rep_k.append((output["key_embeddings"][lay][i, :, :]*mask[i, :, :]).sum(dim=0)/nb_tokens[i, :])
                            rep_v.append((output["value_embeddings"][lay][i, :, :]*mask[i, :, :]).sum(dim=0)/nb_tokens[i, :])

                    for lay in range(len(output["hidden_states"])):
                        if ~mean_calc:
                            rep_emb.append(output["hidden_states"][lay][i, 0, :])
                        else:
                            rep_emb.append((output["hidden_states"][lay][i, :, :]*mask[i, :, :]).sum(dim=0)/nb_tokens[i, :])

                    res[f"class_{class_rep}"] = {
                        "key": rep_k,
                        "value": rep_v,
                        "emb": rep_emb
                    }

                    class_rep_found = True
                    break
            if class_rep_found:
                break
    return res


def process_data_frame(df: DataFrame, ids: np.ndarray) -> DataFrame:
    """ Process the data frame of the similarity to provide readeable outputs with boxplots

    Args:
        df (DataFrame): the dataframe we want to proceed
        ids (np.ndarray): the real ids of our values

    Returns (DataFrame):
    """
    cols = df.columns.values  # columns names
    buff_1 = np.concatenate([df[j].values for j in cols])  # concatenante all the cosines
    buff_2 = np.concatenate(
        [np.repeat(c.split("_")[1], df.shape[0]) for c in cols])  # to which value did we compute the sim

    res = pd.DataFrame({
        "cos_sim": buff_1,  # the similarity
        "class_comp": buff_2,  # with which class did we compute the similarity
        "real_class": np.concatenate([ids for _ in cols])
    })

    return res
