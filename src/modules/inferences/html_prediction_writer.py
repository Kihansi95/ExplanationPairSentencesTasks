import json
import os
from os import path

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
import pandas as pd
import pytorch_lightning as pl

from modules import log, map_dict2list, recursive_tensor2list, rescale
from modules.const import InputType
from modules.inferences.writable_interface import WritableInterface

class SingleSequenceHighlighter:
	
	def __init__(self, writable):
		"""Implement logics of how to extract tokens from entry and how to highlight tokens
		"""
		self.tokens = None
		self.writable = writable
	
	def rescale_maps(self, result):
		"""Rescale value map
		
		Parameters
		----------
		value_map : list
			List of values to be rescaled
			
		Returns
		-------
			Rescaled value map
		"""
		
		result['a_hat'] = rescale(result['a_hat'], result['padding_mask'])
		result['a_heu'] = rescale(torch.exp(result['heuristic']), result['padding_mask'])
		
		return result
	
	def extract_tokens(self, entry):
		"""Get tokens from entry
		
		Parameters
		----------
		entry : dict
			Entry
		
		Returns
		-------
			List of tokens
		"""
		self.tokens = self.writable.writing_tokens(entry)
		
	def highlight_tokens(self, value_map: list) -> str:
		"""Generate HTML string highlighting tokens from extract tokens. Be sure to call extract_tokens first.
		
		Parameters
		----------
		value_map : list
			List of values to be highlighted
			
		Returns
		-------
			HTML string
		"""
		assert self.tokens is not None, "Call `extract_tokens` first"
		assert isinstance(value_map, list), "value_map must be a list of token"
		assert len(self.tokens) == len(value_map), "Length of tokens and value_map must be the same"
		
		return "".join([f'<mark v="{v}">{t} </mark>' if v > 0 else t+' ' for t, v in zip(self.tokens, value_map)])
		
		
class DualSequenceHighlighter(SingleSequenceHighlighter):
	
	def __init__(self, writable):
		super(DualSequenceHighlighter, self).__init__(writable)
	
	def rescale_maps(self, result):
		"""Rescale value map

		Parameters
		----------
		value_map : list
			List of values to be rescaled

		Returns
		-------
			Rescaled value map
		"""
		result['a_heu'] = {}
		for side in result['padding_mask'].keys():
			result['a_hat'][side] = rescale(result['a_hat'][side], result['padding_mask'][side])
			result['a_heu'][side] = rescale(torch.exp(result['heuristic'][side]), result['padding_mask'][side])
		
		return result
	
	def highlight_tokens(self, value_map: dict) -> str:
		"""Generate HTML string highlighting tokens from extract tokens. Be sure to call extract_tokens first.

		Parameters
		----------
		value_map : list
			List of values to be highlighted

		Returns
		-------
			HTML string
		"""
		assert self.tokens is not None, "Call `extract_tokens` first"
		assert isinstance(value_map, dict), "value_map must be a dictionary of {'premise', 'hypothesis'}"
		assert len(self.tokens) == len(value_map), "Length of tokens and value_map must be the same"
		
		html = ""
		for side, maps in value_map.items():
			html += f"<b>{side.capitalize()}:</b> "
			html += "".join([f'<mark v="{v}">{t} </mark>' if v > 0 else t+' ' for t, v in zip(self.tokens[side], maps)])
			html += "<br/>"
		
		return html

def format_single_tokens(tokens):
	"""Format tokens to be written in HTML
	
	Parameters
	----------
	tokens : list
		List of tokens
	
	Returns
	-------
		Formatted tokens
	"""
	return " ".join(tokens)
	
class HtmlPredictionWriter(BasePredictionWriter):
	
	TEMPLATE_PATH =	path.join(path.dirname(__file__), 'single_sequence_template.html')
	CONTENT_TAG = '{{ content }}'
	
	def __init__(self,
				 output_dir,
				 dm: WritableInterface = None,
				 fname='inference',
				 **kwargs):
		"""Write prediction result in parquet. The writer is called only during prediction phrase

		Parameters
		----------
		output_dir : str
			Where to store resulted file
		"""
		super().__init__(**kwargs)
		self.output_dir = output_dir
		self.fname = fname
		self.dm = dm
		if dm.input_type == InputType.SINGLE:
			self.highlighter = SingleSequenceHighlighter(dm)
		elif dm.input_type == InputType.DUAL:
			self.highlighter = DualSequenceHighlighter(dm)
		else:
			raise ValueError(f'Unknown input type: {dm.input_type}')
		
	def __write_row_html(self, template, data):
		"""Write a row of HTML table
		
		Parameters
		----------
		template : str
			HTML template
		data : dict
			Data to be written in row
		
		Returns
		-------
			HTML row
		"""
		content = []
		for entry in data:
			
			self.highlighter.extract_tokens(entry)
			attention_map = self.highlighter.highlight_tokens(entry['a_hat'])
			annotation_map = self.highlighter.highlight_tokens(entry['a_true'])
			heuristic_map = self.highlighter.highlight_tokens(entry['a_heu'])
				
			row = f"""<tr>
				<td>{entry['id']}</td>
				<td>{attention_map}</td>
				<td>{entry["y_hat"]}</td>
				<td>{entry["y_true"]}</td>
				<td>{annotation_map}</td>
				<td>{heuristic_map}</td>
				</tr>
			"""
			content.append(row)
		
		content_str = "\n".join(content)
		filled_html = template.replace(self.CONTENT_TAG, content_str)
		
		return filled_html
	
	def write_on_batch_end(
			self,
			trainer: pl.Trainer,
			model_module: pl.LightningModule,
			prediction,
			batch_indices,
			batch,
			batch_idx: int,
			dataloader_idx: int):
		
		# Define batch file path
		batch_fpath = path.join(self.output_dir, 'batches', f'{self.fname}.html')
		
		# Fusion prediction and batch
		result = {**prediction, **batch}
		
		# Rescale all type of explanation maps
		result = self.highlighter.rescale_maps(result)
		
		# Convert tensor to list, therefore remove all padding tokens
		result = recursive_tensor2list(result)
		
		# Alternative for converting into dataframe:
		data = self.dm.format_predict(result)
		data = map_dict2list(data)
		
		# Load template html file. If the file is written, load the file instead
		if not path.exists(batch_fpath):
			with open(self.TEMPLATE_PATH, 'r') as f:
				template = f.read()
		else:
			with open(batch_fpath, 'r') as f:
				template = f.read()
		
		# fill into the template
		template = self.__write_row_html(template, data)
		
		# Write to file
		os.makedirs(path.join(self.output_dir, 'batches'), exist_ok=True)
		with open(batch_fpath, 'w') as f:
			f.write(template)
	
	def assemble_batch(self):
		"""Assemble batch predictions into single file

		Returns
		-------
			results: pandas.DataFrame
		"""
		if not path.exists(path.join(self.output_dir, 'batches')):
			log.warning(f'Batch-output folder not found in {self.output_dir}. No assemble')
			return None
		
		batch_folder = path.join(self.output_dir, 'batches')
		with open(path.join(batch_folder, f'{self.fname}.html'), 'r') as f:
			content = f.read()
		
		# remove content tag
		content.replace(self.CONTENT_TAG, '')
		
		inference_path = path.join(self.output_dir, f'batch_{self.fname}.html')
		with open(inference_path, 'w', encoding='utf-8') as f:
			f.write(content)
		
		log.info(f'Finished assembling inference files from {batch_folder}.')
		log.info(f'Inferences are stored in {inference_path}')
		
		