import os
from os import path


def gen_ckp(num_layers: int = 1,
            num_heads: int = 1,
            dataset: int = 1,
            log_path: str = "logs",
            run: int = 0,
            model: str = "pur_attention",
            name: str = "PureAttention"):
    """
    Args:
        num_layers: number of layers
        num_heads: number of heads
        dataset: the used dataset
        log_path: fpath for the logs
        run : the number of the run
        model: the type of model we use (default pur_attention i.e pur attention based model)
        name: name of the experience (check the pure attention in the src folder)

    Returns:
        ckp : the checkpoint directory to the corresponding experience
        hparams : fpath for the hparams of the model
    """
    ckp = path.join(log_path, name, f"{model}_{dataset}_l={num_layers}_h={num_heads}_run={run}", "checkpoints",
                    "best.ckpt")

    hparams = path.join(log_path, name, f"{model}_{dataset}_l={num_layers}_h={num_heads}_run={run}",
                    "hparams.yaml")
    return ckp, hparams
