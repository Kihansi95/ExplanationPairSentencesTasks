import argparse

# import self as self
import json

import torch

import pytorch_lightning as pl
from DataModules.SnliDM import ESNLIDataModule
from DataModules.HateXplainDM import HateXPlainDataModule

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
from torchmetrics import AUROC
from torchmetrics import AveragePrecision
from torchmetrics import MetricCollection

from transformers import BertModel
from transformers import BertTokenizer

import os
from os import path

from pytorch_lightning import callbacks as cb

from logger import log, init_logging

tk = BertTokenizer.from_pretrained('bert-base-uncased')

#############
### model ###
#############

# constants for numerical stabilities
EPS = 1e-16
INF = 1e30


def L2D(batch):
    # convert list of dict to dict of list
    if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}
    return {k: [row[k] for row in batch] for k in batch[0]}


class BertNliRegu(pl.LightningModule):

    def __init__(self, freeze_bert=False, criterion=nn.CrossEntropyLoss(), lr=5e-5,
                 exp: bool = False):
        super().__init__()
        self.exp = exp

        self.lr = lr

        # bert layer
        # the bert layer will return the layer will return the attention weights
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_attentions=True,  # return the attention
                                              output_hidden_states=True  # return the hidden states of the models
                                              )

        # classifier head
        self.classifier = nn.Sequential(  # fully connected layer
            nn.Linear(in_features=768, out_features=3)
        )

        self.criterion = criterion

        # metrics
        self.acc = nn.ModuleDict({
            'TRAIN': Accuracy(num_classes=3),
            'VAL': Accuracy(num_classes=3),
            'TEST': Accuracy(num_classes=3)
        })

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # don't save any tensor with gradient, conflict in multiprocessing
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        cls_token = output.last_hidden_state[:, 0, :]

        # the logits are the weights before the softmax.
        logits = self.classifier(cls_token)

        return {"logits": logits, "outputs": output}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer

    #######################
    ### steps functions ###
    #######################

    def step(self, batch, batch_idx):
        # the batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_masks"]
        labels = batch["labels"]
        # forward of the model
        buff = self.forward(input_ids, attention_mask)
        logits = buff["logits"]
        outputs = buff["outputs"]
        # the loss
        loss = self.criterion(logits, labels)

        # the probabilities
        class_pred = torch.softmax(logits, dim=1)

        return {'loss': loss, 'preds': class_pred, 'target': labels}

    def step_end(self, output, stage: str):
        step_acc = self.acc[stage](output['preds'], output['target'])
        if stage == "VAL":
            # for the EarlyStopping
            epoch_bool = True
        else:
            epoch_bool = False
        self.log(f"{stage}_/loss", output['loss'], on_step=True, on_epoch=epoch_bool, logger=True)
        self.log(f"{stage}_/acc", step_acc, on_step=True, on_epoch=epoch_bool, logger=True, prog_bar=True)

    def end_epoch(self, stage):
        d = dict()
        d[f"{stage}_acc"] = round(self.acc[stage].compute().item(), 4)
        log.info(f"Epoch : {self.current_epoch} >> {stage}_metrics >> {d}")

    ####################
    ### the training ###
    ####################

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx)

    def training_step_end(self, output):
        self.step_end(output=output, stage="TRAIN")

    def on_train_epoch_end(self):
        return self.end_epoch(stage="TRAIN")

    ######################
    ### the validation ###
    ######################
    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx)

    def validation_step_end(self, output):
        self.step_end(output=output, stage="VAL")

    def on_validation_epoch_end(self):
        return self.end_epoch(stage="VAL")

    ################
    ### the test ###
    ################
    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch, batch_idx)

    def test_step_end(self, output):
        test_acc = self.acc["TEST"](output['preds'], output['target'])
        self.log("hp_/loss", output["loss"], on_step=False, on_epoch=True, logger=True)
        self.log("hp_/acc", test_acc, on_step=False, on_epoch=True, logger=True)


def get_num_workers() -> int:
    '''
    Get maximum logical workers that a machine has
    Args:
        default (int): default value

    Returns:
        maximum workers number
    '''
    if hasattr(os, 'sched_getaffinity'):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass

    num_workers = os.cpu_count()
    return num_workers if num_workers is not None else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # .cache folder >> the folder where everything will be saved
    cache = path.join(os.getcwd(), '.cache')

    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)

    # what model we should use the default is 1 >> the one created in this file
    parser.add_argument('-t', '--model_type', type=int, default=1)

    # default datadir >> ./.cache/dataset >> cache for our datamodule.
    parser.add_argument('-d', '--data_dir', default=os.path.join(cache, "HateXPlainDataSet"))

    # log_dir for the logger
    parser.add_argument('-s', '--log_dir', default=path.join(cache, 'logs'))

    parser.add_argument('-n', '--nb_data', type=int, default=-1)
    parser.add_argument('-mn', '--model_name')

    # config to distinguish experimentations
    parser.add_argument('--exp', action='store_true')

    # save in [args.log_dir]/[args.experiments]/[args.version] allow good tensorboard
    parser.add_argument('--experiment', type=str, default='test')
    parser.add_argument('--version', type=str, default='0.0')

    # config for cluster distribution
    parser.add_argument('--num_workers', type=int,
                        default=get_num_workers())  # auto select appropriate cores in machine
    parser.add_argument('--accelerator', type=str, default='auto')  # auto select GPU if exists

    # config for the regularization
    parser.add_argument('--reg_mul', type=float, default=0)  # the regularize terms
    parser.add_argument('--lrate', type=float, default=5e-5)  # the learning rate for the training part

    parser.add_argument('--dataset', type=str, default="esnli")

    args = parser.parse_args()

    if args.exp:
        init_logging(color=not args.exp, cache_path=os.path.join(args.log_dir, args.experiment, args.version),
                     oar_id=f"RegMul={args.reg_mul}")
    else:
        init_logging()

    # Summary information
    log.info(f'>>> Arguments: {json.dumps(vars(args), indent=4)}')

    # load the data for the training part
    if args.dataset == "esnli":
        dm = ESNLIDataModule(cache=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
                             nb_data=args.nb_data)
    if args.dataset == "hatexplain":
        dm = HateXPlainDataModule(cache_path=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
                                  nb_data=args.nb_data)

    dm.prepare_data()

    model = None
    if args.model_type == 1:
        model = BertNliRegu(criterion=nn.CrossEntropyLoss(), lr=args.lrate, exp=args.exp)

    ######################
    ### trainer config ###
    ######################

    # set the direction to visualize the logs of the training
    # the visualization will be done with tensorboard.
    logger = TensorBoardLogger(save_dir=args.log_dir,  # the main log folder
                               name=args.experiment,  # name of the log >> related to the name of the model we use
                               version=args.version,  # version of the log
                               default_hp_metric=False  # deactivate hp_metric on tensorboard visualization
                               )
    # logger = TensorBoardLogger(name=args.log_dir, save_dir=log_dir + '/')

    # call back
    early_stopping = cb.EarlyStopping(monitor="VAL_/loss", patience=5, verbose=args.exp, mode='min')
    model_checkpoint = cb.ModelCheckpoint(filename='best',
                                          monitor="VAL_/loss",
                                          mode='min',  # save the minimum val_loss
                                          )

    trainer = pl.Trainer(max_epochs=args.epoch,
                         accelerator=args.accelerator,  # auto use gpu
                         enable_progress_bar=not args.exp,  # hide progress bar in experimentation
                         log_every_n_steps=1,
                         default_root_dir=args.log_dir,
                         logger=logger,
                         callbacks=[early_stopping, model_checkpoint],
                         detect_anomaly=not args.exp)

    #############################
    ### training of the model ###
    #############################
    dm.setup(stage='fit')
    trainer.fit(model, datamodule=dm)

    dm.setup(stage='test')
    performance = trainer.test(ckpt_path='best', datamodule=dm)
    log.info(f"performance of the model : {performance[0]}")
    log.info('Training finished')
