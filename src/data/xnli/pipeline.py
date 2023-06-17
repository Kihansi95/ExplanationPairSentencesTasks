import os
from os import path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from data.xnli.dataset import FrXNLI, _EXTRACTED_FILES
from modules import log

class PretransformedFrXNLI(FrXNLI):
	
	def __init__(self,
	             transformations: List[Dict],
	             prefix: str = '',
	             split: str = 'train',
	             root: str = path.join(os.getcwd(), '.cache'),
	             n_data: int = -1):
		
		# assert
		super(PretransformedFrXNLI, self).__init__(split=split, root=root)
		
		root = self.root(root)
	
		if len(prefix) > 0 and prefix[-1] != '.': prefix += '.'
		self.json_path = path.join(root, 'pretransformed.' + prefix +  _EXTRACTED_FILES[split])
		
		# Check if the preprocessed dataset exists
		if path.exists(self.json_path):
			self.data = pd.read_json(self.json_path)
			log.info(f"Loaded preprocessed dataset from {self.json_path}")
		
		else:
			
			# Check that the input data has the expected structure
			for transform in transformations:
				
				# Check that the keys are the expected ones
				assert all(key in transform for key in ["output_name", "input_name", "transformation"]), f"Invalid data format: {transform}"
				assert isinstance(transform["output_name"], str), f"Invalid data format: {transform}"
				assert isinstance(transform["input_name"], list), f"Invalid data format: {transform}"
				
				output_name = transform["output_name"]
				input_names = transform["input_name"]
				fn = transform["transformation"]
				
				log.debug('Applying transformation: ' + output_name)
				
				if len(input_names) == 1:
					tqdm.pandas(desc=output_name)
					self.data[output_name] = self.data[input_names].progress_apply(fn)
				else:
					input_columns = [self.data[input_name] for input_name in input_names]
					output_column = fn(*input_columns)
					self.data[output_name] = pd.Series(output_column)
			
			# save the preprocessed dataset
			self.data.to_json(self.json_path, force_ascii=False)
			log.info(f"Saved preprocessed dataset to {self.json_path}")
		
		# reduce dataset proportionally for each class
		self.sample_data(n_data, 'label')

# Test the pretransformed dataset with a simple transformation
if __name__ == '__main__':
	
	# Define a simple transformation
	def concat(x, y):
		return x + y
	
	transformations = [
		{
			"output_name": "concatenated",
			"input_name": ["premise.text", "hypothesis.text"],
			"transformation": concat
		}
	]
	
	# Create the pretransformed dataset
	dataset = PretransformedFrXNLI(
		transformations=transformations,
		split='train',
		root=path.join(os.getcwd(), '..', '..', '.cache', 'dataset'),
		n_data=10)
	
	# Check that the transformation has been applied
	print(dataset.data.head())