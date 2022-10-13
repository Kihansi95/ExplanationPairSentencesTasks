import os
import shutil

import pandas as pd
from os import path

import spacy

import numpy as np
from spacy.tokens import Doc

from data.yelp_hat.dataset import YelpHat, DATASET_NAME
from data.yelp_hat.utils import yelp_hat_ham, yelp_hat_token
from modules.logger import log
	
def _reformat_dataframe(data: pd.DataFrame, spacy_model=None, lemma:bool=True, lower:bool=True):
	
	if spacy_model is None: spacy_model=spacy.load('en_core_web_sm')
	
	# Binarizing human attention map
	for idx in range(3):
		data[f'ham_{idx}'] = data[f'ham_html_{idx}'].apply(lambda x: yelp_hat_ham(x, spacy_model)).apply(lambda x: np.array(x))
		
	# Pre tokenize
	data['text_tokens'] = data['ham_html_0'].apply(lambda x: yelp_hat_token(x, spacy_model, lemma, lower))
	
	# Drop incoherent attention maps samples
	data_drop = data[(data['ham_0'].str.len() == data['ham_1'].str.len()) & (data['ham_1'].str.len() == data['ham_2'].str.len())].reset_index(drop=True)
	n_drop = len(data) - len(data_drop)
	
	if n_drop > 0:
		log.warning(f'Drop {n_drop} samples because HAMs are not compatibles')
		data = data_drop
	
	# Synthetize the rationale
	data['ham'] = data.apply(lambda row: ((row['ham_0'] + row['ham_1'] + row['ham_2']) / 3 >= 0.5).astype(int), axis=1)
	data['cam'] = data.apply(lambda row: row['ham_0'] * row['ham_1'] * row['ham_2'], axis=1)
	data['sam'] = data.apply(lambda row: ((row['ham_0'] + row['ham_1'] + row['ham_2']) > 0).astype(int), axis=1)
	
	
	# convert numpy into list:
	for column in ['ham_0', 'ham_1', 'ham_2', 'ham', 'cam', 'sam']:
		data[column] = data[column].apply(lambda x: x.tolist())
	
	if (data.text_tokens.str.len() != data.ham_0.str.len()).any():
		mismatch_index = data.index[data.text_tokens.str.len() != data.ham_0.str.len()]
		raise ValueError(f'Tokens and Rationale dimension mismatch at: {mismatch_index}')
		
	# heuristic
	text_tokens = data['text_tokens'].tolist()
	
	## make pos filter
	docs = [Doc(spacy_model.vocab, words=sent) for sent in text_tokens]
	tokenized_docs = list(spacy_model.pipe(docs))
	pos = [[tk.pos_ for tk in d] for d in tokenized_docs]
	data['pos_tag'] = pd.Series(pos)
	pos_filter = [[tk.pos_ in ['NOUN', 'VERB', 'ADJ'] for tk in d] for d in docs]
	stop_filter = [[not tk.is_stop for tk in d] for d in docs]
	mask = [pos_ and stop_ for pos_, stop_ in zip(pos_filter, stop_filter)]

	## Count words
	token_freq = dict()
	flatten_token = [tk for sent in text_tokens for tk in sent]
	flatten_rationale = [r for sent in text_tokens for r in sent]
	
	for t, r in zip(flatten_token, flatten_rationale):
		if r: token_freq[t] = token_freq.get(t, 0) + 1
	
	total_freq = sum(token_freq.values())
	token_freq = {k: v / total_freq for k, v in token_freq.items()}
	
	## build heuristics
	heuristics = []
	for sent_tokens, sent_mask in zip(text_tokens, mask):
		heuris_map = [token_freq.get(tk, 0) for tk in sent_tokens]
		heuris_map = [h * float(m) for h, m in zip(heuris_map, sent_mask)]
		sum_heuris = max(sum(heuris_map), 1)
		heuris_map = [h/sum_heuris for h, m in zip(heuris_map, sent_mask)]
		heuristics.append(heuris_map)
	
	data['heuristic'] = pd.Series(heuristics)
	
	# put back rationale and token into list
	#data['rationale'] = data.rationale.apply(lambda x: x.tolist())
	#data['post_tokens'] = data.post_tokens.apply(lambda x: x.tolist())
	
	return data

class SpacyPretokenizeYelpHat(YelpHat):
	
	def __init__(self, split: str = 'yelp', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1, spacy_model=None, lemma:bool=True, lower:bool=True):
		
		super(SpacyPretokenizeYelpHat, self).__init__(split=split, root=root, n_data=n_data)
		root = self.root(root)
		
		fname, fext = os.path.splitext(self.parquet_path)
		fprep = 'pretokenized' + ('_lower' if lower else '') + ('_lemma' if lemma else '')
		self.parquet_path = path.join(root, f'{fname}.{fprep}{fext}')
		
		if path.exists(self.parquet_path):
			self.data = pd.read_parquet(self.parquet_path)
			for col in ['text_tokens', 'ham_0', 'ham_1', 'ham_2', 'ham', 'cam', 'sam']:
				self.data[col] = self.data[col].apply(lambda x: x.tolist())
		else:
			self.data = _reformat_dataframe(self.data, spacy_model, lemma, lower)
			self.data.to_parquet(self.parquet_path)
			log.info(f'Save yelp subset {split} at: {self.parquet_path}')
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label))  # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subset).reset_index(drop=True)
	
	@classmethod
	def root(cls, root):
		return path.join(root, DATASET_NAME)


if __name__ == '__main__':
	# Unit test
	
	from torch.utils.data import DataLoader
	
	cache_path = path.join(os.getcwd(), '..', '..', '..', '.cache', 'dataset')
	
	trainset = SpacyPretokenizeYelpHat(root=cache_path, split='train')
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	b = next(iter(train_loader))
	print('train batch:')
	print(b)
