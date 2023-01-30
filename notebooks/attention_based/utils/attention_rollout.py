import os.path
from tqdm import tqdm
import torch
from torch import Tensor
from sklearn import metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout(model, dm, verbose: bool = True):
    """
    This function is made to compute the attention rollout and compute some properties on the attention.

    Args:
        model: an attention based model
        dm: the datamodule to treat
        verbose (bool) : verbose contrôle

    Returns:

    """
    test_dataloader = dm.test_dataloader()
    n_layer = model.num_layers

    rollout_map = []
    a_true_map = []
    padding_map = []

    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader), total=int(dm.n_data / dm.batch_size))

        for id_batch, batch in pbar:

            pbar.set_description("proceed the cosine map")

            y_true = batch["y_true"].to(model.device)
            ids = batch["token_ids"].to(model.device)
            padding_mask = batch["padding_mask"].bool().to(model.device)
            T = (~padding_mask).float().sum(dim=-1).unsqueeze(dim=-1).repeat(1, padding_mask.shape[1])

            temp = (~padding_mask)
            temp[:, 0] = False
            temp = temp.flatten()
            padding_map.append(temp)

            # the output
            output = model(ids=ids, mask=padding_mask)

            # the rollout
            attention = torch.stack(output["attn_weights"], dim=1)  # shape [N, Layer, T, T]
            #values = torch.stack(output["value_embeddings"])  # shape [N, Layer, T, d]

            rollout_buff = attention[:, 0, :, :]  # we take the attention at the layer 0
            if n_layer > 1:
                for k in range(1, n_layer):
                    rollout_buff = torch.bmm(attention[:, k, :, :], rollout_buff)

            rollout_buff = rollout_buff.sum(dim=1)  # sum over the lines to compute the attention rollout.
            rollout_buff = rollout_buff / T
            rollout_map.append(rollout_buff.flatten())

            # a_true map
            a_true = batch["a_true"].to(model.device)
            a_true = a_true.flatten()
            a_true_map.append(a_true)

        # compute the different metrics.
        res = {}
        padding_map = torch.concat(padding_map)
        a_true_map = torch.concat(a_true_map)[padding_map]
        rollout_map = torch.concat(rollout_map)[padding_map]

        if verbose:
            print(f"test passed : {a_true_map.shape}")
            print()

        fpr, tpr, thresholds = metrics.roc_curve(a_true_map.int().cpu().numpy(),
                                                 rollout_map.cpu().numpy(),
                                                 pos_label=1)
        temp = metrics.auc(fpr, tpr)
        res["AUC - cos"] = temp
        res["rollout_values"] = rollout_map.cpu().numpy() # return the map.

    return res
