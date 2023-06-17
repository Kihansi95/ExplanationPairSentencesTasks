import json
import pickle
import sys
from collections import Counter
from os import path

import pandas as pd
import pytorch_lightning as pl
import spacy
import torch
import torchtext.transforms as T
from torch.utils.data import DataLoader
from torchtext.vocab import vocab as build_vocab

from tqdm import tqdm

from data.transforms import LemmaLowerTokenizerTransform
from data.xnli.pipeline import PretransformedFrXNLI
from modules import log, env
from modules.const import SpecToken


class FrXNLIDM(pl.LightningDataModule):
    name = 'Fr-XNLI'

    def __init__(self, cache_path, batch_size=8, num_workers=0, n_data=-1, shuffle=True):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        # Dataset already tokenized
        self.n_data = n_data
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.input_type = PretransformedFrXNLI.INPUT
        self.num_class = PretransformedFrXNLI.NUM_CLASS
        self.LABEL_ITOS = PretransformedFrXNLI.LABEL_ITOS

        spacy_model = spacy.load('fr_core_news_sm')
        tokenizer_transform = LemmaLowerTokenizerTransform(spacy_model)
        # heuristic_transform = HeuristicTransform(
        #     vectors=GloVe(cache=path.join(cache_path, '..', '.vector_cache')),
        #     spacy_model=spacy_model,
        #     cache=cache_path
        # )
        
        # Transformations preapplied to the dataset
        self.transformations = [
            { 'output_name': 'premise.tokens', 'input_name': ['premise.text'], 'transformation': tokenizer_transform},
            { 'output_name': 'hypothesis.tokens', 'input_name': ['hypothesis.text'], 'transformation': tokenizer_transform},
        ]

    def prepare_data(self):
        # called only on 1 GPU

        # download_dataset()
        dataset_path = PretransformedFrXNLI.root(self.cache_path)
        
        self.vocab_path = path.join(dataset_path, f'vocab.pt')

        for split in ['train', 'val', 'test']:
            PretransformedFrXNLI.download_format_dataset(dataset_path, split)

        # build_vocab()
        if not path.exists(self.vocab_path):
            
            # build vocab if not exist
            log.info(f'Vocab not found at {self.vocab_path}, building vocab...')
            
            # Make words frequency dictionary
            freq_dict_path = path.join(dataset_path, 'frequency_lemma_lower.json')
            if not path.exists(freq_dict_path):
                
                train_set = PretransformedFrXNLI(
                    transformations=self.transformations,
                    root=self.cache_path,
                    split='train',
                    n_data=self.n_data
                )
                
                word_counter = Counter()
                # list of tokenized sentences
                token_list = train_set.data['premise.tokens'].tolist() + train_set.data['hypothesis.tokens'].tolist()
                # flatten into list of tokens
                token_list = [token for tokens in token_list for token in tokens]
                for token in tqdm(token_list, desc='Building frequency dictionary', total=len(token_list), unit='sents', file=sys.stdout, disable=env.disable_tqdm):
                    word_counter.update(token)
                
                # sort by frequency descending
                word_frequency = word_counter.most_common()
                word_frequency = {word: cnt for word, cnt in word_frequency}
                
                with open(freq_dict_path, 'w') as f:
                    json.dump(word_frequency, f)
                    
            else :
                with open(freq_dict_path, 'r') as f:
                    word_frequency = json.load(f)
            
            vocab = build_vocab(word_frequency, specials=[SpecToken.PAD, SpecToken.UNK], min_freq=5, special_first=True)
            vocab.set_default_index(vocab[SpecToken.UNK])
            torch.save(vocab, self.vocab_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol to speed things up
            self.vocab = vocab
            log.info(f'Vocabulary is saved at {self.vocab_path}')
            
        else:
            self.vocab = torch.load(self.vocab_path)
            log.info(f'Loaded vocab at {self.vocab_path}')
        
        log.info(f'Vocab size: {len(self.vocab)}')

        # Clean cache
        PretransformedFrXNLI.clean_cache(root=self.cache_path)

        # Transformations applied during collation
        self.text_transform = T.Sequential(
            T.VocabTransform(self.vocab),
            T.ToTensor(padding_value=self.vocab[SpecToken.PAD])
        )

        self.label_transform = T.Sequential(
            T.LabelToIndex(['neutre', 'implication', 'contradiction']),
            T.ToTensor()
        )

    def setup(self, stage: str = None):
        dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data, transformations=self.transformations)

        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train_set = PretransformedFrXNLI(split='train', **dataset_kwargs)
            self.val_set = PretransformedFrXNLI(split='val', **dataset_kwargs)

        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_set = PretransformedFrXNLI(split='test', **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()

    def format_predict(self, prediction: pd.DataFrame):
        
        return prediction

    ## ======= PRIVATE SECTIONS ======= ##

    def collate(self, batch):
        # prepare batch of data for dataloader
        b = self.list2dict(batch)

        b.update({
            'premise.ids': self.text_transform(b['premise.tokens']),
            'hypothesis.ids': self.text_transform(b['hypothesis.tokens']),
        })
        
        if 'label' in b:
            b.update({'y_true': self.label_transform(b['label'])})
        
        b['padding_mask'] = {
            'premise': b['premise.ids'] == self.vocab[SpecToken.PAD],
            'hypothesis': b['hypothesis.ids'] == self.vocab[SpecToken.PAD],
        }
        
        return b

    def list2dict(self, batch):
        # convert list of dict to dict of list

        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
        return {k: [row[k] for row in batch] for k in batch[0]}