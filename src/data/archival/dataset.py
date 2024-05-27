import os
from os import path

import pandas as pd
from sklearn.model_selection import train_test_split

from modules import log
from modules.const import InputType

DATASET_NAME = 'archival'
NUM_CLASS = 2
INPUT = InputType.DUAL
SEED = 42

# Root taken from tools/nli.
RAW_FILE = path.join(os.getcwd(), '.cache', 'archival_nli.json')

_EXTRACTED_FILES = {
	'train': 'train.json',
	'val': 'val.json',
	'test': 'test.json',
	'full': 'archival_nli.json'
}

_TEST_SPLIT = .15  # 15% of original dataset
_VAL_SPLIT = .15  # 15% of original dataset


def download_format_dataset(root: str, split: str, version: str):
	"""
	Download and reformat dataset
	Args:
		root (str): cache folder where to find the dataset.
		split (str): among train, val, test
		raw_path (int): parqueet file generated by notebook
	"""
	if path.basename(root) != DATASET_NAME:
		root = path.join(root, DATASET_NAME)
    root = path.join(root, version)  # add version to the root
	os.makedirs(root, exist_ok=True)
	# make a subdata set for dev purpose
	json_path = path.join(root, _EXTRACTED_FILES[split])
	
	if not path.exists(json_path):
		
		json_path = path.join(root, _EXTRACTED_FILES['full'])
		full_data = pd.read_json(json_path, encoding='utf-8')
		
		# split into train-val-test equally for classes
        train_val, test = train_test_split(full_data, test_size=_TEST_SPLIT, stratify=full_data['label'],
                                           random_state=SEED)
        train, val = train_test_split(train_val, test_size=_VAL_SPLIT / (1 - _TEST_SPLIT), stratify=train_val['label'],
                                      random_state=SEED)
		
		for split_set, split in zip([train, val, test], ['train', 'val', 'test']):
			split_set.reset_index(drop=True, inplace=True)
			split_path = path.join(root, _EXTRACTED_FILES[split])
			with open(split_path, 'w', encoding='utf-8') as f:
				split_set.to_json(f, force_ascii=False)
				log.info(f'Save {split} set at: {split_path}')
	
	return json_path


class ArchivalNLI(MapDataPipe):
	NUM_CLASS = NUM_CLASS
	INPUT = INPUT
    
    def __init__(self, split: str = 'train', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1,
                 version: str = None):
		
		if split not in _EXTRACTED_FILES.keys():
			self.json_path = split
			self.split = 'predict'
        # 	raise ArgumentError(f'split argument {split} doesnt exist for ArchivalNLI')
		
		else:
			root = self.root(root)
			self.split = split
			if version is None:
				version = self.get_version(root)
			
			# download and prepare csv file if not exist
			self.json_path = download_format_dataset(root, split, version=version)
		
		self.data = pd.read_json(self.json_path, encoding='utf-8')
		
		log.info(f'Load dataset from {self.json_path}')
		
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
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
	def get_version(cls, root):
		"""
		Get the newest version created in the folder. Version folder name should be started by v

		Parameters
		----------
		root :

		Returns
		-------
		str
			name of the newest folder

		"""
		versions = [d for d in os.listdir(root) if path.isdir(path.join(root, d)) and d != '.DS_Store' and d[0] == 'v']
		versions.sort(key=lambda x: os.stat(os.path.join(root, x)).st_mtime)
		return versions[-1]
	
	@classmethod
	def download_format_dataset(cls, root, split='test', version=None):
		if version is None:
			version = cls.get_version(root)
		return download_format_dataset(root, split, version)
	
	@classmethod
	def clean_cache(cls, root):
		pass
	
	def __str__(self):
		description = {
			'name': DATASET_NAME,
			'split': self.split,
			'cache': self.json_path,
			'#data': len(self.data),
			'More': 'Use dataset.data.describe()'
		}
		return str(description)
	
	def describe(self):
		return self.data.describe()