import os
import shutil

import datasets
import pandas as pd
from os import path

from datasets import load_dataset

from torch.utils.data import MapDataPipe

from tqdm import tqdm

from data import ArgumentError
from modules.const import InputType
from modules.logger import log

DATASET_NAME = 'hotpot_qa'
INPUT = InputType.SINGLE

_EXTRACTED_FILES = {
	'train': 'train.csv',
	'val': 'val.csv',
	'test': 'test.csv',
}

_SPLIT_MAPPING = {
	'train': 'train',
	'val': 'train',
	'test': 'validation'
}

_VAL_SPLIT = 0.3


def download_format_dataset(root: str, split: str, n_data: int = -1, subset='distractor'):
	"""
	Download and reformat dataset of eSNLI
	Args:
		root (str): cache folder where to find the dataset.
		split (str): among train, val, test
		n_data (int): maximum data to load. -1 to load entire data
	"""
	
	if path.basename(root) != DATASET_NAME:
		root = path.join(root, DATASET_NAME)
	# raise TypeError(f'Please concatenate root folder with dataset name: `{DATASET_NAME}`')
	
	# make a subdata set for dev purpose
	
	# download the dataset
	original_set = load_dataset(DATASET_NAME, subset, split=_SPLIT_MAPPING[split], cache_dir=root)
	if split in ['train', 'val']:
		spliting_set = original_set.train_test_split(test_size=_VAL_SPLIT, shuffle=False)
		
		for file_split, func_split in zip(['train', 'val'], spliting_set):
			# print('func_split', func_split)
			df = _reformat_csv(spliting_set[func_split], file_split)
			csv_path = path.join(root, _EXTRACTED_FILES[file_split])
			df.to_csv(csv_path, index=False, encoding='utf-8')
			log.info(f'Preprocess {file_split} and saved at {csv_path}')
	else:
		# treat test set
		df = _reformat_csv(original_set, split)
		csv_path = path.join(root, _EXTRACTED_FILES[split])
		df.to_csv(csv_path, index=False, encoding='utf-8')
		log.info(f'Preprocess {split} and saved at {csv_path}')
	
	return csv_path
	
	
def _reformat_csv(dataset: datasets.Dataset, split):
		
	data_dict = {'question': list(), 'context': list(),
				 'facts': list(),
				 'support': list(),
				 'answer': list(),
				 'level': list()}
	
	warn_msg = list()
	for row in tqdm(dataset, total=len(dataset), desc=f'Formating {split}'):
		
		_id = row['id']
		question = row['question']
		
		# find ids of context:
		facts = row['supporting_facts']
		context = row['context']
		context_length = [len(c) for c in context['sentences']]
		context_text = [' '.join(c) for c in context['sentences']]
		
		
		facts_mask = [[False] * cl for cl in context_length]
		# TODO find better way to use compacted facts_mask
		facts = context['sentences'].copy()
		
		for title, id_sent in zip(facts['title'], facts['sent_id']):
			id_title = context['title'].index(title)
			if id_sent >= context_length[id_title]:
				warn_msg += [f'id sent not valid in [title={id_title},sent={id_sent}], id={_id}']
			else:
				# TODO here too
				facts[id_title][id_sent] = '*' + facts[id_title][id_sent] + '*'
				facts_mask[id_title][id_sent] = True
		
		for text, length, highlighted_fact, mask in zip(context_text, context_length, facts, facts_mask):
			data_dict['question'] += [question]
			data_dict['context'] += [text]
			data_dict['highlighted_facts'] += [highlighted_fact]
			data_dict['facts'] += [mask]
			data_dict['support'] += [any(mask)]
			data_dict['answer'] += [row['answer']]
			data_dict['level'] += [row['level']]
	
	log.warning('\n'.join(warn_msg))

	return pd.DataFrame(data_dict)


class HotpotNLIDataset(MapDataPipe):
	
	def __init__(self, split: str = 'train',
	             root: str = path.join(os.getcwd(), '.cache'),
	             n_data: int = -1,
	             subset: str = 'distractor'):
		
		# assert
		if split not in _EXTRACTED_FILES.keys():
			raise ArgumentError(f'split argument {split} doesnt exist for {type(self).__name__}')
		
		root = self.root(root)
		self.split = split
		self.csv_path = path.join(root, _EXTRACTED_FILES[split])
		
		# download and prepare csv file if not exist
		download_format_dataset(root=root, split=split, n_data=n_data, subset=subset)
		
		coltype = {'question': str, 'support': bool, 'answer': str,
		           'level': str}
		
		col_convert = {'facts': pd.eval, 'context': pd.eval}
		
		# load the csv file to data
		self.data = pd.read_csv(self.csv_path, dtype=coltype, converters=col_convert)
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			subset = [pd.DataFrame()] * len(_unique_label)
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label))  # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subset).reset_index(drop=True)
	
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
	def download_format_dataset(cls, root, split, n_data):
		return download_format_dataset(root, split, n_data)


if __name__ == '__main__':
	# Unit test
	
	from torch.utils.data import DataLoader, Dataset
	
	cache_path = path.join('/Users', 'dunguyen', 'Projects', 'nlp', 'src', '_out', 'dataset')
	
	# To load the 3 at same time:
	# trainset, valset, testset = ESNLIDataPipe(root=cache_path)
	trainset, valset = HotpotNLIDataset(root=cache_path, split=('train', 'valid'))
	
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
