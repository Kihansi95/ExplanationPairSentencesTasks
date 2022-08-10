import pytorch_lightning as pl
import pickle
import os
from os import path
import shutil

import datasets
import pandas as pd
import torch

from datasets import load_dataset
from torch.utils.data import MapDataPipe, DataLoader
import torchtext.transforms as T
from transformers import BertTokenizer

import DataModules.transforms as t
import numpy as np

DATASET_NAME = 'hatexplain'

_EXTRACTED_FILES = {
    'train': 'train.csv',
    'val': 'val.csv',
    'test': 'test.csv',
}


class ArgumentError(ValueError):
    pass


def download_format_dataset(root: str, split: str):
    """
    Download and reformat dataset of eSNLI
    Args:
        root (str): cache folder where to find the dataset.
        split (str): among train, val, test
        n_data (int): maximum data to load. -1 to load entire data
    """

    if path.basename(root) != DATASET_NAME:
        root = path.join(root, DATASET_NAME)

    csv_path = path.join(root, _EXTRACTED_FILES[split])
    if path.exists(csv_path):
        return csv_path

    huggingface_split = 'validation' if split == 'val' else split

    # download the dataset
    dataset = load_dataset(DATASET_NAME, split=huggingface_split, cache_dir=root)

    # Correct the example
    df = _reformat_csv(dataset, split)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    return csv_path


def clean_cache(root: str):
    shutil.rmtree(path.join(root, 'downloads'), ignore_errors=True)
    for fname in os.listdir(root):
        if fname.endswith('.lock'): os.remove(os.path.join(root, fname))


######################################
### construct and reformat the csv ###
######################################

tk = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_row(row):
    post_tokens = row[0]
    rationale = row[2]

    pos_tk_ids = []
    rat = []

    for id_w, w in enumerate(post_tokens):
        t = tk(w)
        # remove special tokens
        ids = t.input_ids
        _ = ids.pop(0)
        _ = ids.pop(-1)
        pos_tk_ids += ids

        if rationale[id_w] == 1:
            rat += [1] * len(ids)
        else:
            rat += [0] * len(ids)

    return {"bert_ids": pos_tk_ids,
            "bert_post_tokens": tk.convert_ids_to_tokens(pos_tk_ids),
            "bert_rationale": rat}


def get_bert_ids(row):
    return row[3]["bert_ids"]


def get_bert_post_tok(row):
    return row[3]["bert_post_tokens"]


def get_bert_rationale(row):
    return row[3]["bert_rationale"]


def _reformat_csv(dataset: datasets.Dataset, split):
    df = dataset.to_pandas()

    # Correct 1 example in train set
    if split == 'train':
        rationales = df.loc[1997, 'rationales']
        L = len(df.loc[1997, 'post_tokens'])
        rationales = [r[:L] for r in rationales]
        df.loc[1997, 'rationales'] = rationales

    # gold label = most voted label
    df['label'] = df.annotators.apply(lambda x: np.bincount(x['label']).argmax())
    # rationale = average rationaled then binarize by 0.5 threshold
    df['rationale'] = df.rationales.apply(
        lambda x: (np.mean([r.astype(float) for r in x], axis=0) >= 0.5).astype(int) if len(x) > 0 else x)

    # put back label into text
    int2str = ['hatespeech', 'normal', 'offensive']  # huggingface's label
    df['label'] = df.label.apply(lambda x: int2str[x]).astype('category')

    # make rationale for negative example, for padding coherent
    df['len_tokens'] = df.post_tokens.str.len()
    df['rationale'] = df.apply(
        lambda row: np.zeros(row['len_tokens'], dtype=np.int32) if len(row['rationale']) == 0 else row['rationale'],
        axis=1)
    df = df.drop(columns='len_tokens')

    # put back rationale and token into list
    df['rationale'] = df.rationale.apply(lambda x: x.tolist())
    df['post_tokens'] = df.post_tokens.apply(lambda x: x.tolist())

    df = df.drop(columns=['annotators', 'rationales', 'id'])

    df["buff"] = df.apply(tokenize_row, axis=1)
    df["bert_ids"] = df.apply(get_bert_ids, axis=1)
    df["bert_post_tokens"] = df.apply(get_bert_post_tok, axis=1)
    df["bert_rationale"] = df.apply(get_bert_rationale, axis=1)

    df = df.drop(columns=['buff'])

    return df


class HateXPlain(MapDataPipe):

    def __init__(self, split: str = 'train',
                 root: str = path.join(os.getcwd(), '.cache'),
                 n_data: int = -1):

        # assert
        if split not in _EXTRACTED_FILES.keys():
            raise ArgumentError(f'split argument {split} doesnt exist for {type(self).__name__}')

        root = self.root(root)
        self.split = split
        # download and prepare csv file if not exist
        self.csv_path = download_format_dataset(root=root, split=split)
        col_type = {'label': 'category'}
        col_convert = {'post_tokens': pd.eval, 'rationale': pd.eval, 'bert_ids': pd.eval, 'bert_rationale': pd.eval}

        # load the csv file to data
        self.data = pd.read_csv(self.csv_path, dtype=col_type, converters=col_convert)

        # if n_data activated, reduce the dataset equally for each class
        if n_data > 0:
            _unique_label = self.data['label'].unique()

            subsets = [
                self.data[self.data['label'] == label]  # slice at each label
                    .head(n_data // len(_unique_label))  # get the top n_data/3
                for label in _unique_label
            ]
            self.data = pd.concat(subsets).reset_index(drop=True)

    def __getitem__(self, index: int):

        if index >= len(self): raise IndexError

        sample = self.data.loc[index].to_dict()
        return sample

    def __len__(self):
        """
        Denotes the total number of samples
        Returns: int
        """
        return len(self.data)

    @classmethod
    def root(cls, root):
        return path.join(root, DATASET_NAME)

    @classmethod
    def download_format_dataset(cls, root, split):
        return download_format_dataset(root, split)

    @classmethod
    def clean_cache(cls, root):
        return clean_cache(root)


class HateXPlainDataModule(pl.LightningDataModule):

    def __init__(self, cache_path, batch_size=8, num_workers=0, nb_data=-1):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        # Dataset already tokenized
        self.dataset = {'train': None, 'val': None, 'test': None}
        self.n_data = nb_data
        self.num_workers = num_workers

    def prepare_data(self):
        # called only on 1 GPU
        # download_dataset()
        dataset_path = HateXPlain.root(self.cache_path)
        for split in ['train', 'val', 'test']:
            HateXPlain.download_format_dataset(dataset_path, split)
        # Clean cache
        HateXPlain.clean_cache(root=self.cache_path)

        self.label_transform = T.Sequential(
            T.LabelToIndex(['normal', 'hatespeech', 'offensive']),
            T.ToTensor()
        )
        self.tensor_transform = T.Sequential(
            T.ToTensor(padding_value=0),  # put all the element of the batch to the same size
            T.PadTransform(max_length=150, pad_value=0)
        )

    def setup(self, stage: str = None):
        dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data)

        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train_set = HateXPlain(split='train', **dataset_kwargs)
            self.val_set = HateXPlain(split='val', **dataset_kwargs)

        if stage == 'test' or stage is None:
            self.test_set = HateXPlain(split='test', **dataset_kwargs)

        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    ## ======= PRIVATE SECTIONS ======= ##
    def collate(self, batch):
        # a bert collate
        batch = self.list2dict(batch)
        input_ids = self.tensor_transform(batch["bert_ids"])
        attention_masks = (input_ids == 0).type(torch.uint8)
        annotations = self.tensor_transform(batch["bert_rationale"])
        labels = self.label_transform(batch["label"])

        b = {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
            "annotations": annotations
        }

        return b

    def list2dict(self, batch):
        # convert list of dict to dict of list

        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
        return {k: [row[k] for row in batch] for k in batch[0]}
