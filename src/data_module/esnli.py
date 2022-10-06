import pickle
import sys

import pytorch_lightning as pl
import spacy
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator, GloVe
import torchtext.transforms as T
from tqdm import tqdm
from os import path

from data.esnli.legacy_transform_dataset import PretransformedESNLI
from data.esnli.transforms import HighlightTransform, HeuristicTransform
from data.transforms import LemmaLowerTokenizerTransform
from data_module.constant import *
from modules import env
from modules.logger import log

class ESNLIDM(pl.LightningDataModule):
	
	def __init__(self, cache_path, batch_size=8, num_workers=0, n_data=-1):
		super().__init__()
		self.cache_path = cache_path
		self.batch_size = batch_size
		# Dataset already tokenized
		self.n_data = n_data
		self.num_workers = num_workers
		self.num_class = PretransformedESNLI.NUM_CLASS
		self.input_type = PretransformedESNLI.INPUT
		
		spacy_model = spacy.load('en_core_web_sm')
		tokenizer_transform = LemmaLowerTokenizerTransform(spacy_model)
		hl_transform = T.Sequential( tokenizer_transform, HighlightTransform())
		heuristic_transform = HeuristicTransform(
			vectors=GloVe(cache=path.join(cache_path, '..', '.vector_cache')),
			spacy_model=spacy_model,
			normalize='log_softmax'
		)
		
		self.transformations = {
			'premise': tokenizer_transform,
			'hypothesis': tokenizer_transform,
			'highlight_premise': hl_transform,
			'highlight_hypothesis': hl_transform,
			'heuristic': heuristic_transform,
		}
		self.rename_columns = {
			'premise': 'premise_tokens',
			'hypothesis': 'hypothesis_tokens',
			'highlight_premise': 'premise_rationale',
			'highlight_hypothesis': 'hypothesis_rationale',
			'heuristic': 'heuristic'
		}
	
	def prepare_data(self):
		# called only on 1 GPU
		
		# download_dataset()
		dataset_path = PretransformedESNLI.root(self.cache_path)
		vocab_path = path.join(dataset_path, f'vocab.pt')
		
		for split in ['train', 'val', 'test']:
			PretransformedESNLI.download_format_dataset(dataset_path, split)
		
		# build_vocab()
		if not path.exists(vocab_path):
			
			# return a single list of tokens
			def flatten_token(batch):
				return [token for sentence in batch['premise_tokens'] + batch['hypothesis_tokens'] for token in sentence]
			
			train_set = PretransformedESNLI(transformations=self.transformations, column_name=self.rename_columns, root=self.cache_path, split='train', n_data=self.n_data)
			
			# build vocab from train set
			dp = train_set.batch(self.batch_size).map(self.list2dict).map(flatten_token)
			
			# Build vocabulary from iterator. We don't know yet how long does it take
			iter_tokens = tqdm(iter(dp), desc='Building vocabulary', total=len(dp), unit='sents', file=sys.stdout, disable=env.disable_tqdm)
			if env.disable_tqdm: log.info(f'Building vocabulary')
			vocab = build_vocab_from_iterator(iterator=iter_tokens, specials=[PAD_TOK, UNK_TOK])
			# vocab = build_vocab_from_iterator(iter(doc for doc in train_set['post_tokens']), specials=[PAD_TOK, UNK_TOK])
			vocab.set_default_index(vocab[UNK_TOK])
			
			# Announce where we save the vocabulary
			torch.save(vocab, vocab_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol to speed things up
			iter_tokens.set_postfix({'path': vocab_path})
			if env.disable_tqdm: log.info(f'Vocabulary is saved at {vocab_path}')
			iter_tokens.close()
			self.vocab = vocab
		else:
			self.vocab = torch.load(vocab_path)
			log.info(f'Loaded vocab at {vocab_path}')
		
		log.info(f'Vocab size: {len(self.vocab)}')
		
		# Clean cache
		PretransformedESNLI.clean_cache(root=self.cache_path)
		
		# predefined processing mapper for setup
		self.text_transform = T.Sequential(
			T.VocabTransform(self.vocab),
			T.ToTensor(padding_value=self.vocab[PAD_TOK])
		)
		
		self.rationale_transform = T.Sequential(
			T.ToTensor(padding_value=0)
		)
		
		self.label_transform = T.Sequential(
			T.LabelToIndex(['neutral', 'entailment', 'contradiction']),
			T.ToTensor()
		)
	
	def setup(self, stage: str = None):
		dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data,
		                      transformations=self.transformations, column_name=self.rename_columns,)
		
		# called on every GPU
		if stage == 'fit' or stage is None:
			self.train_set = PretransformedESNLI(split='train', **dataset_kwargs)
			self.val_set = PretransformedESNLI(split='val', **dataset_kwargs)
		
		if stage == 'test' or stage is None:
			self.test_set = PretransformedESNLI(split='test', **dataset_kwargs)
	
	def train_dataloader(self):
		return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
	
	def val_dataloader(self):
		return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
	
	def test_dataloader(self):
		return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
	
	## ======= PRIVATE SECTIONS ======= ##
	def collate(self, batch):
		# prepare batch of data for dataloader
		batch = self.list2dict(batch)
		
		b = {
			'premise_ids': self.text_transform(batch['premise_tokens']),
			'hypothesis_ids': self.text_transform(batch['hypothesis_tokens']),
			'a_true': {
				'premise': self.rationale_transform(batch['premise_rationale']),
				'hypothesis': self.rationale_transform(batch['hypothesis_rationale']),
			},
			'y_true': self.label_transform(batch['label'])
		}
		
		b['padding_mask'] = {
			'premise': b['premise_ids'] == self.vocab[PAD_TOK],
			'hypothesis': b['hypothesis_ids'] == self.vocab[PAD_TOK],
		}
		
		return b
	
	def list2dict(self, batch):
		# convert list of dict to dict of list
		
		if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
		return {k: [row[k] for row in batch] for k in batch[0]}