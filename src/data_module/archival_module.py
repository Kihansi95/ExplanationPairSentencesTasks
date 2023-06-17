import json
import pickle
import sys
from os import path

import pandas as pd
import pytorch_lightning as pl
import torch
import torchtext
import torchtext.transforms as T
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from data.archival.dataset import ArchivalNLI
from modules import env
from modules.const import SpecToken
from modules.logger import log


class ArchivalNLIDM(pl.LightningDataModule):
    
    name = 'archival_nli_module'

    def __init__(self, cache_path, batch_size=8, num_workers=0, n_data=-1, shuffle=True, predict_path=None, version=None):
        """Process/Preprocess ArchivalNLI dataset
        
        Parameters
        ----------
        cache_path : str
            Passed to dataset. Path to general dataset (Ex. `.../datasets`)
        batch_size : int
            Passed to dataloader. batch size to query in dataloader Default : 8
        num_workers : int, optional
            Passed to dataloader. Normally auto determined by script. 0 to run in the main process. Default : 0
        n_data : int
            Passed to dataset. Maximum data size to train, used in debug. `n_data=-1` to load entire dataset.  Default : -1.
        shuffle : bool, optional
            If false, the train set won't be shuffled. Default : True
        predict_path : str
            If given, dataloader with load a custom dataset to make prediction.
        version : str
            Passed to dataset. `version = None` will get the last Archival version. Default : None
        """
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        # Dataset already tokenized
        self.n_data = n_data
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.num_class = ArchivalNLI.NUM_CLASS
        self.input_type = ArchivalNLI.INPUT
        self.predict_path = predict_path
        self.version = version

    def prepare_data(self):
        # called only on 1 GPU

        # download_dataset()
        dataset_path = ArchivalNLI.root(self.cache_path)
        self.vocab_path = path.join(dataset_path, self.version, 'vocab.pt')
        self.vocab_dict_path = path.join(dataset_path, self.version, 'frequency_token_norm.json')

        for split in ['train', 'val', 'test']:
            ArchivalNLI.download_format_dataset(dataset_path, split, version=self.version)

        # build_vocab()
        if not path.exists(self.vocab_path):
            if path.exists(self.vocab_dict_path):
                log.debug(f'Build vocab from dictionary : {self.vocab_dict_path}')
                with open(self.vocab_dict_path, 'r') as f:
                    vocab_dict = json.load(f)
                    vocab = torchtext.vocab.vocab(vocab_dict, min_freq=2, specials=[SpecToken.PAD, SpecToken.UNK, SpecToken.ENT_PER, SpecToken.ENT_LOC, SpecToken.ENT_MISC, SpecToken.ENT_ORG,])
                    vocab.set_default_index(vocab[SpecToken.UNK])
            else:
                log.debug(f'Build vocab from dictionary : {self.vocab_dict_path}')
                # return a single list of tokens
                def flatten_token(batch):
                    return [token for sent in batch['premise.norm'] + batch['hypothesis.norm'] for token in sent]
    
                train_set = ArchivalNLI(root=self.cache_path, split='full', n_data=self.n_data, version=self.version)
    
                # build vocab from train set
                dp = train_set.batch(self.batch_size).map(self.list2dict).map(flatten_token)
    
                # Build vocabulary from iterator. We don't know yet how long does it take
                iter_tokens = tqdm(iter(dp), desc='Building vocabulary', total=len(dp), unit='sents', file=sys.stdout, disable=env.disable_tqdm)
                if env.disable_tqdm: log.info(f'Building vocabulary')
                vocab = build_vocab_from_iterator(iterator=iter_tokens, min_freq=2, specials=[SpecToken.PAD, SpecToken.UNK, SpecToken.ENT_PER, SpecToken.ENT_LOC, SpecToken.ENT_MISC, SpecToken.ENT_ORG,])
                vocab.set_default_index(vocab[SpecToken.UNK])
                iter_tokens.set_postfix({'fpath': self.vocab_path})
                iter_tokens.close()
                
            # Announce where we save the vocabulary
            torch.save(vocab, self.vocab_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol to speed things up
            if env.disable_tqdm: log.info(f'Vocabulary is saved at {self.vocab_path}')
            
            self.vocab = vocab
        else:
            self.vocab = torch.load(self.vocab_path)
            log.info(f'Loaded vocab at {self.vocab_path}')

        log.info(f'Vocab size: {len(self.vocab)}')

        # Clean cache
        ArchivalNLI.clean_cache(root=self.cache_path)

        # predefined processing mapper for setup
        self.text_transform = T.Sequential(
            T.VocabTransform(self.vocab),
            T.ToTensor(padding_value=self.vocab[SpecToken.PAD])
        )

        self.label_transform = T.Sequential(
            T.LabelToIndex(['neutral', 'entailment']),
            T.ToTensor()
        )

    def setup(self, stage: str = None):
        dataset_kwargs = dict(
            root=self.cache_path,
            n_data=self.n_data,
            version=self.version
        )

        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train_set = ArchivalNLI(split='train', **dataset_kwargs)
            self.val_set = ArchivalNLI(split='val', **dataset_kwargs)

        if stage == 'test' or stage is None:
            self.test_set = ArchivalNLI(split='test', **dataset_kwargs)
            
        if stage == 'predict' or stage is None:
            if self.predict_path is None:
                self.predict_set = ArchivalNLI(split='test', **dataset_kwargs)
            else:
                self.predict_set = ArchivalNLI(split=self.predict_path, **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def format_predict(self, prediction: pd.DataFrame):
        import numpy as np
        
        # remove padding using mask
        def remove_mask(row):
            for side in ['premise', 'hypothesis']:
                padding_mask = np.array(row[f'padding_mask.{side}'])
        
                a_hat = np.array(row[f'a_hat.{side}'])
                a_hat = a_hat[~padding_mask]
                row[f'a_hat.{side}'] = a_hat.tolist()
        
                ids = np.array(row[f'{side}_ids'])
                ids = ids[~padding_mask]
                row[f'{side}_ids'] = ids.tolist()
            return row

        if 'padding_mask.premise' in prediction.columns:
            inference_sentences = prediction.apply(remove_mask, axis=1)
            inference_sentences.drop(columns=['padding_mask.premise', 'padding_mask.hypothesis'], inplace=True)
        
        return prediction

    ## ======= PRIVATE SECTIONS ======= ##

    def collate(self, batch):
        # prepare batch of data for dataloader
        b = self.list2dict(batch)

        b.update({
            'premise.ids': self.text_transform(b['premise.norm']),
            'hypothesis.ids': self.text_transform(b['hypothesis.norm']),
        })
        
        if 'label' in b:
            b['y_true'] = self.label_transform(b['label'])

        b['padding_mask'] = {
            'premise': b['premise.ids'] == self.vocab[SpecToken.PAD],
            'hypothesis': b['hypothesis.ids'] == self.vocab[SpecToken.PAD],
        }

        return b

    def list2dict(self, batch):
        # convert list of dict to dict of list

        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
        return {k: [row[k] for row in batch] for k in batch[0]}

