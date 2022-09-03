import os
import shutil

import datasets
import pandas as pd
from os import path

from datasets import load_dataset

from torch.utils.data import MapDataPipe

import numpy as np

from data import ArgumentError
from modules.logger import log

DATASET_NAME = 'hatexplain'
NUM_CLASS = 3

_EXTRACTED_FILES = {
	'train': 'train.parquet',
	'val': 'val.parquet',
	'test': 'test.parquet',
}


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
	
	parquet_path = path.join(root, _EXTRACTED_FILES[split])
	if path.exists(parquet_path):
		return parquet_path
	
	huggingface_split = 'validation' if split == 'val' else split
	
	# download the dataset
	dataset = load_dataset(DATASET_NAME, split=huggingface_split, cache_dir=root)
	
	# Correct the example
	df = _reformat_dataframe(dataset, split)
	df.to_parquet(parquet_path)
	log.info(f'Process {split} and saved at {parquet_path}')
	
	return parquet_path
	

def clean_cache(root: str):
	shutil.rmtree(path.join(root, 'downloads'), ignore_errors=True)
	for fname in os.listdir(root):
		if fname.endswith('.lock'): os.remove(os.path.join(root, fname))
	
	
def _reformat_dataframe(dataset: datasets.Dataset, split):
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
	df['rationale'] = df.rationales.apply(lambda x: (np.mean([r.astype(float) for r in x], axis=0) >= 0.5).astype(int) if len(x) > 0 else x)
	
	# put back label into text
	int2str = ['hatespeech', 'normal', 'offensive']  # huggingface's label
	df['label'] = df.label.apply(lambda x: int2str[x]).astype('category')
	
	# make rationale for negative example, for padding coherent
	df['len_tokens'] = df.post_tokens.str.len()
	df['rationale'] = df.apply(lambda row: np.zeros(row['len_tokens'], dtype=np.int32) if len(row['rationale']) == 0 else row['rationale'], axis=1)
	df = df.drop(columns='len_tokens')
	
	# put back rationale and token into list
	df['rationale'] = df.rationale.apply(lambda x: x.tolist())
	df['post_tokens'] = df.post_tokens.apply(lambda x: x.tolist())
	
	df = df.drop(columns=['annotators', 'rationales', 'id'])
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
		
		# download and prepare parquet file if not exist
		self.parquet_path = download_format_dataset(root=root, split=split)
		
		# load the parquet file to data
		self.data = pd.read_parquet(self.parquet_path)
		
		self.data['post_tokens'] = self.data['post_tokens'].apply(lambda x: x.tolist())
		self.data['rationale'] = self.data['rationale'].apply(lambda x: x.tolist())
		
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
		"""

		Args:
			index ():

		Returns:

		"""
		
		# Load data and get label
		if index >= len(self): raise IndexError  # meet the end of dataset
		
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

if __name__ == '__main__':
	# Unit test
	
	from torch.utils.data import DataLoader, Dataset
	
	cache_path = path.join('/Users', 'dunguyen', 'Projects', 'nlp', 'src', '_out', 'dataset')
	
	# To load the 3 at same time:
	# trainset, valset, testset = ESNLIDataPipe(root=cache_path)
	trainset, valset = HateXPlain(root=cache_path, split=('train', 'val'))
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	for b in train_loader:
		print('train batch:')
		print(b)
		break
	
	val_loader = DataLoader(valset, batch_size=1, shuffle=False)
	for b in train_loader:
		print('val batch:')
		print(b)
		break
