import os.path
from tqdm import tqdm
import torch
from torch import Tensor

# the metrics will be calculated with sklearn
from sklearn import metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RELU = torch.nn.ReLU()
INF = 1e16


# TODO : verify this function
def compute_rolling_cos(ids: Tensor,
                        cls_states: Tensor,
                        model: torch.nn.Module,
                        padding_idx: int = 0):
    """
    Args:
        ids (Tensor): ids of the different sentences in the batch we treat
        cls_states:
        model:
        padding_idx:

    Returns:

    """

    res = torch.zeros(ids.shape, dtype=float, device=model.device)
    mask = ids != padding_idx

    # for each sentence
    for i in range(ids.shape[0]):

        curr_rep = cls_states[i, :]  # the vector were
        curr_sent = torch.zeros((ids.shape[1], 300), device=model.device)  # the new embeddings of the sentence

        for k in range(ids.shape[1]):
            # we treat the element k in the model.
            if k == 0:
                curr_ids = ids[i, :][[k, k + 1]]
                p = torch.tensor([False, False], device=model.device)
            if k == ids.shape[1]:
                curr_ids = ids[i, :][[k - 1, k]]
                p = torch.tensor([False, False], device=model.device)
            else:
                curr_ids = ids[i, :][[k - 1, k, k + 1]]
                p = torch.tensor([False, False, False], device=model.device)

            output = model(curr_ids.unsqueeze(0), p.unsqueeze(0))
            buff = output["last_hidden_states"][0, :, :].mean(dim=0)
            curr_sent[k, :] = buff

        # compute the cosine

        dot_prod = torch.matmul(curr_sent, curr_rep)
        nms = torch.norm(curr_sent, dim=-1)

        curr_res = dot_prod / (nms * torch.norm(curr_rep))

        # in comment this is the min max approximation
        # but this map doesn't sum to one.
        mx = curr_res[mask[i, :]].max()
        mn = curr_res[mask[i, :]].min()

        if mx > mn:
            curr_res = (curr_res - mn) / (mx - mn)

        res[i, :] = curr_res

    return res


def attention_metrics_res(model, dm, nb_data: int, verbose: bool = True):
    """ attention_metrics_res

    The objective of this function is to compare the attention between the two technics
    cosine and attention based

    Args:
        model: attention based model
        dm: the datamodule which contains the data
        cache (str) : were to save or load the embedding matrix
        nb_data (int) : number of sentences to proceed
        verbose (bool) : should we print some results

    Returns:
        returns result of some plausibility metrics.
    """
    cond = 8

    # proceed the attention
    test_dataloader = dm.test_dataloader()
    nb_it = 0

    attention_map = []
    cosine_map = []
    a_true_map = []
    padding_map = []

    with torch.no_grad():

        pbar = tqdm(enumerate(test_dataloader), total=int(dm.n_data / dm.batch_size))

        for id_batch, batch in pbar:

            pbar.set_description("proceed the cosine map")

            y_true = batch["y_true"].to(model.device)
            ids = batch["token_ids"].to(model.device)[y_true != cond, :]
            padding_mask = batch["padding_mask"].bool().to(model.device)[y_true != cond, :]

            temp = (~padding_mask)
            temp[:, 0] = False
            temp = temp.flatten()
            padding_map.append(temp)

            # a_true map
            a_true = batch["a_true"].to(model.device)[y_true != cond, :]
            a_true = a_true.flatten()
            a_true_map.append(a_true)

            output = model(ids=ids, mask=padding_mask)

            # a_hat map
            attention_tensor = torch.stack(output['attn_weights'], dim=1)  # [N, 1, L, L]
            a_hat = attention_tensor[:, 0, 0, :]  # of size (N, L)
            a_hat = a_hat.flatten()  # flatten the attention map
            attention_map.append(a_hat)

            # cosine map
            cls_tokens = output["cls_tokens"]  # last states for the embeddings
            cos = compute_rolling_cos(ids, cls_tokens, model)
            cos = cos.flatten()  # flatten the cosine map
            cosine_map.append(cos)

            nb_it += ids.shape[0]
            if nb_it >= nb_data:
                print("done !")
                break

        # compute the different metrics.
        res = {}
        padding_map = torch.concat(padding_map)

        attention_map = torch.concat(attention_map)[padding_map]
        cosine_map = torch.concat(cosine_map)[padding_map]
        a_true_map = torch.concat(a_true_map)[padding_map]

        assert a_true_map.shape == cosine_map.shape, "error : cos"
        assert a_true_map.shape == attention_map.shape, "error : attention"

        if verbose:
            print(f"test passed : {a_true_map.shape}")
            print()

        fpr, tpr, thresholds = metrics.roc_curve(a_true_map.int().cpu().numpy(),
                                                 cosine_map.cpu().numpy(),
                                                 pos_label=1)
        temp = metrics.auc(fpr, tpr)
        res["AUC - cos"] = temp

        fpr, tpr, thresholds = metrics.roc_curve(a_true_map.int().cpu().numpy(),
                                                 attention_map.cpu().numpy(),
                                                 pos_label=1)
        temp = metrics.auc(fpr, tpr)
        res["AUC - attention"] = temp

    return res
