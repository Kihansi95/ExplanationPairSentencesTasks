import os

import pandas as pd
from os import path

import spacy

from data.esnli.dataset import ESNLI, _EXTRACTED_FILES
from modules.logger import log


class PretransformedESNLI(ESNLI):
	
	def __init__(self, transformations: dict, column_name: dict, split: str = 'train', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1):
		"""

		Args:
			split       (str):
			cache_path  (str):
			n           (int): max of data to be loaded
			shuffle     (bool): shuffle if load limited data
						If n is precised and shuffle = True, dataset will sample n datas.
						If n is precised and shuffle = False, dataset will take only n first datas.
		"""
		
		# assert
		super(PretransformedESNLI, self).__init__(split=split, root=root, n_data=n_data)
		assert transformations.keys() == column_name.keys(), f'Incoherent item between transforms and column_name. \n\tTransforms: {list(transformations.keys())} \n\tColumns: {list(column_name.keys())}'
		
		root = self.root(root)
		
		fname, fext = os.path.splitext(_EXTRACTED_FILES[split])
		self.parquet_path = path.join(root, f'{fname}.pretransformed{fext}')
		
		
		if path.exists(self.parquet_path):
			# load the cache file to data if file exist
			self.data = pd.read_parquet(self.parquet_path)
			for column in ['premise_tokens', 'hypothesis_tokens', 'premise_rationale', 'hypothesis_rationale', 'premise_heuristic', 'hypothesis_heuristic']:
				self.data[column] = self.data[column].apply(lambda x: x.tolist())
			
		else:
			# pretransform if not exist, then save to cache
			heuristic_transform = transformations.pop('heuristic')
			
			for column, transform in transformations.items():
				new_column = column_name[column]
				log.debug(f'Transforming: {column} -> {new_column}')
				self.data[new_column] = pd.Series(transform(self.data[column]))
				
			heuristic_col = column_name['heuristic']
			result = heuristic_transform(premise=self.data['premise'], hypothesis=self.data['hypothesis'])
			
			for side in ['premise', 'hypothesis']:
				score = result[side]
				mask = result[f'{side}_mask']
				# get length (# of token) for each sentence
				sent_length = mask.sum(dim=1).tolist()
				# flatten the score by masking padding scores, then resplit by sentence length
				unpad_heuristic = score[mask.bool()].split(sent_length)
				# transform into list to save into cache
				unpad_heuristic = [h.tolist() for h in unpad_heuristic]
				self.data[f'{side}_{heuristic_col}'] = pd.Series(unpad_heuristic)
				
				# assert columns
			self.data.to_parquet(self.parquet_path)
			log.info(f'Save pretransform {split} eSNLI at: {self.parquet_path}')
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label)) # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subset).reset_index(drop=True)
	
		
	
if __name__=='__main__':
	# Unit test
	
	from torch.utils.data import DataLoader
	
	cache_path = path.join(os.getcwd(), '..', '..', '..', '.cache', 'dataset')
	
	# To load the 3 at same time:
	# trainset, valset, testset = ESNLIDataPipe(root=cache_path)
	trainset = PretransformedESNLI(root=cache_path, split='train')
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	
	b = next(iter(train_loader))
	print('train batch:')
	print(b)
