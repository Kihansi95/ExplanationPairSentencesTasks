import json
import os
from os import path
import shutil

import torch

from modules.logger import log

INF = 1e30 # Infinity

def rescale(attention: torch.Tensor, mask: torch.Tensor):
	v_max = torch.max(attention + mask.float() * -INF, dim=1, keepdim=True).values
	v_min = torch.min(attention + mask.float() * INF, dim=1, keepdim=True).values
	v_min[v_min == v_max] = 0.
	rescale_attention = (attention - v_min)/(v_max - v_min)
	rescale_attention[mask] = 0.
	return rescale_attention

def hightlight_txt(txt, weights):
	"""
	Build an HTML of text along its weights.
	Args:
		txt:
		weights:

	Returns: str
	Examples:
		```python
		from IPython.core.display import display, HTML
		highlighted_text = hightlight_txt(lemma1[0], a1v2)
		display(HTML(highlighted_text))
		```
	"""
	max_alpha = 0.8

	highlighted_text = ''
	w_min, w_max = torch.min(weights), torch.max(weights)
	w_norm = (weights - w_min)/(w_max - w_min)

	for i in range(len(txt)):
		highlighted_text += '<span style="background-color:rgba(135,206,250,' \
							+ str(float(w_norm[i]) / max_alpha) + ');">' \
							+ txt[i] + '</span> '

	return highlighted_text


def report_score(scores: dict, logger, score_dir=None) -> None:
	"""
	Report scores into score.json and logger
	:param scores: dictionary that has reported scores
	:type scores: dict
	:param logger: Tensorboard logger. Report into hyperparameters
	:type logger: TensorBoardLogger
	:param score_dir: directory to find score.json
	:type score_dir: str
	:return: None
	:rtype: None
	"""
	
	# remove 'TEST/' from score dicts:
	scores = [{k.replace('TEST/', ''): v for k, v in s.items()} for s in scores]
	
	for idx, score in enumerate(scores):
		log.info(score)
		logger.log_metrics(score)
		
		if score_dir is not None:
			os.makedirs(score_dir, exist_ok=True)
			src = path.join(logger.log_dir, 'hparams.yaml')
			dst = path.join(score_dir, 'hparams.yaml')
			shutil.copy2(src, dst)
			
		score_path = path.join(score_dir or logger.log_dir, f'score{"" if idx == 0 else "_" + str(idx)}.json')
		
		with open(score_path, 'w') as fp:
			json.dump(score, fp, indent='\t')
			log.info(f'Score is saved at {score_path}')
		
		
		