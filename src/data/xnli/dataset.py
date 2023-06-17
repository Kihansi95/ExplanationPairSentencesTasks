import os
import shutil

import pandas as pd
from os import path

from datasets import load_dataset
from torchtext.data.datasets_utils import _create_dataset_directory
from torchtext.utils import download_from_url, extract_archive

from torch.utils.data import MapDataPipe

from data import ArgumentError
from modules.const import InputType, SEED
from modules.logger import log

DATASET_NAME = path.join('xnli')
NUM_CLASS = 3
INPUT = InputType.DUAL

_EXTRACTED_FILES = {
    'train': 'train.json',
    'val': 'dev.json',
    'test': 'test.json',
}

_HF_LABEL_ITOS = ['implication', 'neutre', 'contradiction']
_LABEL_ITOS = ['neutre', 'implication', 'contradiction']
_LABEL_STOI = {label: index for index, label in enumerate(_LABEL_ITOS)}


def download_format_dataset(root:str, split:str):
	"""Download and reformat dataset of XNLI/fr
	
	Parameters
	----------
	root : str
		cache folder where to find the dataset. If the dataset is not found, it will be downloaded in this folder.
	split : str
		train, val, test

	Returns
	-------

	"""
	
	if path.join(DATASET_NAME, 'fr') not in root:
		root = path.join(root, DATASET_NAME, 'fr')
	
	# make a subdata set for dev purpose
	json_path = path.join(root, _EXTRACTED_FILES[split])
	if path.exists(json_path):
		return json_path
	
	# download the dataset
	huggingface_split = 'validation' if split == 'val' else split
	dataset = load_dataset(DATASET_NAME, 'fr', split=huggingface_split, cache_dir=path.join(root, '..', '..'))
	
	# reformat file
	df = _reformat_dataframe(dataset.to_pandas())
	df.to_json(json_path, force_ascii=False)
	
	return json_path
	
@_create_dataset_directory(dataset_name=DATASET_NAME)
def clean_up_dataset_cache(root):
	# clean up unnecessary files
	pass

def _reformat_dataframe(data: pd.DataFrame):
		"""
		Remove unecessary columns, rename columns for better understanding. Notice that we also remove extra explanation
		columns.
		Args: data (pandas.DataFrame): Original data given by eSNLI dataset

		Returns:
			(pandas.DataFrame) clean data
		"""
		
		# Convert numeric labels into string label to avoid confusion
		data = data.replace({'label': {index: label for index, label in enumerate(_HF_LABEL_ITOS)}})
		data = data.rename(columns={'premise': 'premise.text', 'hypothesis': 'hypothesis.text'})
		data['label'] = data['label'].astype('category')
		
		return data

class FrXNLI(MapDataPipe):
	
	NUM_CLASS = NUM_CLASS
	INPUT = INPUT
	LABEL_ITOS = _LABEL_ITOS
	
	def __init__(self, split: str = 'train', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1):
		"""XNLI dataset on french language
		
		Parameters
		----------
		split : str
			train, val or test
		root : str
			Root directory of dataset
		n_data : int
			Number of data to load. -1 to load entire data.
		"""
		
		# assert
		assert split in _EXTRACTED_FILES.keys(), f'split argument {split} doesnt exist for eSNLI'
		
		root = self.root(root)
		self.split = split
		self.json_path = path.join(root, _EXTRACTED_FILES[split])
		
		# download and prepare csv file if not exist
		download_format_dataset(root, split)
		
		# load the csv file to data
		self.data = pd.read_json(self.json_path)
		
		# reduce dataset proportionally for each class
		self.sample_data(n_data, 'label')
			
	
	def sample_data(self, n_data, stratify_by='label'):
		"""Sample data with replacement to have equal number of data for each label
		
		Parameters
		----------
		n_data : int
			Number of data to sample
		stratify_by : str
			Column name to stratify by
			
		"""
		# assert
		assert self.data is not None, 'Data is not loaded yet'
		assert stratify_by in self.data.columns, f'Column {stratify_by} doesnt exist in the dataset'

		if n_data < 0 : return
		
		stratify_column = self.data[stratify_by]
		
		# calculate the ratio of each label
		label_ratios = stratify_column.value_counts(normalize=True)
		
		# calculate the weight of each row based on its label
		weights = stratify_column.map(lambda label: 1 / label_ratios[label])
		
		# sample rows with replacement using the calculated weights
		self.data = self.data.sample(n=n_data, weights=weights, replace=True, random_state=SEED).reset_index(drop=True)
	
	def __getitem__(self, index: int):
		
		# Load data
		if index >= len(self): raise IndexError  # meet the end of dataset
		
		sample = self.data.loc[index].to_dict()
		
		return sample
	
	def __len__(self):
		"""Denotes the total number of samples
		
		Returns
		-------
		int
			number of samples
		"""
		return len(self.data)
	
	@classmethod
	def root(cls, root): return path.join(root, DATASET_NAME, 'fr')
	
	@classmethod
	def download_format_dataset(cls, root, split):
		return download_format_dataset(root, split)
	
	@classmethod
	def clean_cache(cls, root):
		return None
		
	
if __name__=='__main__':
	# Unit test
	
	from torch.utils.data import DataLoader

	cache_path = path.join(os.getcwd(), '..', '..', '..', '.cache', 'dataset')
	
	# To load the 3 at same time:
	# trainset, valset, testset = ESNLIDataPipe(root=cache_path)
	trainset = FrXNLI(root=cache_path, split='train')
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	
	b = next(iter(train_loader))
	print('train batch:')
	print(b)
