"""
The objective of this file is to create a new attention map based on the cosine between embeddings in the BERT space
"""
import os.path
from tqdm import tqdm
import torch
from torch import Tensor

# the metrics will be calculated with sklearn
from sklearn import metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RELU = torch.nn.ReLU()
INF = 1e16


def dict_print(d):
    for k in d:
        print(k, " : ", d[k])


###############################################
### Calculation of the new embedding matrix ###
###############################################

def create_embeddings(model, cache: str) -> Tensor:
    """
    Args:
        model: a pur attention based model
        cache : where to load the embedding matrix

    Returns:
        return a tensor of the shape (len(vocab), 300) and save it at the fpath : cache.
    """
    if os.path.exists(cache):
        print(f"load the matrix at the location {cache} ...", end=" ")
        emb_matrix = torch.load(cache, map_location=torch.device('cpu'))
        print("loading done !")
    else:
        assert hasattr(model, "vocab"), "error : the given model doesn't have the vocab argument"
        print("vectors proceeding")

        vocab_itos = model.vocab.get_itos()
        emb_matrix = torch.zeros((len(vocab_itos), 300), device=model.device)

        with torch.no_grad():
            for i in range(len(vocab_itos)):
                # iterate through all the words
                ids = torch.tensor([[model.vocab["<cls>"], i]], device=model.device)
                p = torch.tensor([[False, False]], device=model.device)
                output = model(ids=ids, mask=p)
                emb_matrix[i, :] = output["last_hidden_states"][0, 0, :]

        torch.save(emb_matrix, cache)
        print("done !")

    # return the embedding matrix on a casual device.
    return emb_matrix.cpu()


#####################################
### computation of the cosine map ###
#####################################

def compute_attention_cos_map(ids: Tensor, embedding: torch.nn.Module, cls_states: Tensor,
                              padding_idx: int = 0) -> Tensor:
    """compute_attention_cos_map

    Compute the cosine "attention" for the given batch.

    Args:
        ids (Tensor): tensor of the ids of a sentence (N, L)
        embedding (Module): an embedding layer
        cls_states (Tensor): the cls states of the different sentences (N, d)
        padding_idx (int): the padding idx.

    Returns:
        return the matrix of the cosines.

    """
    res = torch.zeros(ids.shape, dtype=float, device=ids.device)
    new_emb = embedding(ids)  # tensor of shape (N, L, d)
    mask = ids != padding_idx

    # for each sentence
    for i in range(new_emb.shape[0]):
        curr_sent = new_emb[i, :, :]
        curr_rep = cls_states[i, :]  # the vector usefull for the classification of the sentence i

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


###################################################
### computation of the results for the notebook ###
###################################################

def attention_metrics_res(model, dm, cache: str, nb_data: int, verbose: bool = True):
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
    cond = 0
    if model.data == "yelphat":
        # for the yelphat dataset we take all the labels into account.
        cond = 10
    if verbose:
        print(cond)
    emb_matrix = create_embeddings(model, cache)  # load the embedding matrix
    embedding = torch.nn.Embedding.from_pretrained(emb_matrix,
                                                   freeze=True,
                                                   padding_idx=dm.vocab["<pad>"])
    embedding.to(model.device)

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
            ids = batch["tokens"].to(model.device)[y_true != cond, :]
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
            a_hat = attention_tensor[:, model.num_layers - 1, 0, :]  # of size (N, L)
            a_hat = a_hat.flatten()  # flatten the attention map
            attention_map.append(a_hat)

            # cosine map
            cls_tokens = output["cls_tokens"]
            cos = compute_attention_cos_map(ids, embedding, cls_tokens)
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
