import pandas as pd
from os import path


from torch.utils.data import MapDataPipe

class FileDataset(MapDataPipe):
	
	def __init__(self,
	             fpath:str,
	             format:str=None,
	             label_field='label',
	             **kwargs):
		
		if format is None:
			if fpath[-5:] == '.json':
				format = 'json'
			elif fpath[-8:] == '.parquet':
				format = 'parquet'
			elif fpath[-4:] == '.csv':
				format = 'csv'
			else:
				raise NotImplementedError(f'Format of {fpath} unsupported')
		
		if format == 'json':
			self.data = pd.read_json(fpath, **kwargs)
		elif format == 'parquet':
			self.data = pd.read_parquet(fpath, **kwargs)
		elif format == 'csv':
			self.data = pd.read_csv(fpath, **kwargs)
			
		self.ITOS_LABELS = self.data[label_field].unique().tolist()
		self.STOI_LABELS = { label_str: label_idx for label_idx, label_str in enumerate(self.ITOS_LABELS) }
		self.DATASET_NAME = path.splitext(fpath)[0]
		self.NUM_CLASS = len(self.STOI_LABELS)
	
	def __getitem__(self, index: int):
		if index >= len(self): raise IndexError  # meet the end of dataset
		return self.data.loc[index].to_dict()
	
	def __len__(self):
		return len(self.data)
