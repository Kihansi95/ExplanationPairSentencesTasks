import string
from typing import Union

import torch
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab.vectors import Vectors, pretrained_aliases as pretrained


class GoldLabelTransform(Module):
	"""
	Turn eSNLI gold label into corresponding class id
	"""
	
	LABEL_MAPS = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
	
	def __init__(self, label_map: dict = None):
		super(GoldLabelTransform, self).__init__()
		self.lmap_ = label_map if label_map is not None else self.LABEL_MAPS
	
	def forward(self, labels: Union[list, str]):
		if isinstance(labels, str):
			return self.lmap_[labels]
		return [self.lmap_[l] for l in labels]


class HighlightTransform(Module):
	"""
	Turn human highlights into boolean mask, True if it's attention
	"""
	
	def forward(self, highlight: Union[list, str]):
		masks = [list()] * len(highlight)
		for idx, phrase in enumerate(highlight):
			mask = []
			is_highlight = False
			for token in phrase:
				if token == '*':
					is_highlight = not is_highlight
					continue
				mask.append(is_highlight and token not in string.punctuation)
			masks[idx] = mask
		return masks


class HeuristicTransform(Module):
	
	def __init__(self, vectors: str or Vectors, spacy_model, cache=None, normalize: str = None):
		super(HeuristicTransform, self).__init__()
		
		self.N_SOFTMAX = 'softmax'
		self.N_LOG_SOFTMAX = 'log_softmax'
		NORMS = [self.N_SOFTMAX, self.N_LOG_SOFTMAX]
		
		if normalize is not None:
			normalize = normalize.lower()
			assert normalize in NORMS, f'Undefined normalization: {normalize}. Possible values: {NORMS}'
		self.normalize = normalize
		self.sm = spacy_model
		if isinstance(vectors, str):
			vectors = pretrained[vectors](cache=cache)
		self.vectors = vectors
		self.POS_FILTER = ['VERB', 'NOUN', 'ADJ']
		self.INF = 1e30
		self.EPS = 1e-20
	
	def forward(self, premise, hypothesis):
		batch = {'premise': premise, 'hypothesis': hypothesis}
		vectors = {}
		mask = {}
		heuristic = {}
		padding_mask = {}
		
		for side, texts in batch.items():
			docs = list(self.sm.pipe(texts))
			
			padding = [torch.tensor([1] * len(d)) for d in docs]
			
			# POS-tag mask: True on informative tokens
			# pos = [[tk.pos_ for tk in d] for d in docs]
			pos_mask = [torch.tensor([(not tk.is_stop) and (tk.pos_ in self.POS_FILTER) for tk in d]) for d in docs]
			
			for idx in range(len(pos_mask)):
				if (~pos_mask[idx]).all():
					pos_mask[idx] = torch.tensor([not tk.is_stop for tk in docs[idx]])
				if (~pos_mask[idx]).all():
					pos_mask[idx] = torch.tensor([True for _ in docs[idx]])  # Uniform attention
			
			pos_mask = pad_sequence(pos_mask, batch_first=True, padding_value=False)
			padding = pad_sequence(padding, batch_first=True, padding_value=0)
			
			tokens = [[tk.lemma_.lower() for tk in d] for d in docs]
			v = [self.vectors.get_vecs_by_tokens(tk) for tk in tokens]
			v = pad_sequence(v, batch_first=True)
			v_norm = v / (v.norm(dim=2)[:, :, None] + self.EPS)
			
			vectors[side] = v_norm
			mask[side] = pos_mask
			padding_mask[side] = padding
		
		# similarity matrix
		similarity = torch.bmm(vectors['premise'], vectors['hypothesis'].transpose(1, 2))
		
		# apply mask
		for side, dim in zip(batch.keys(), [2, 1]):
			heuristic[side] = similarity.sum(dim).masked_fill_(~mask[side], - self.INF)
		
		# Normalize heuristic
		if self.normalize == self.N_SOFTMAX:
			heuristic = {k: h.softmax(1) for k, h in heuristic.items()}
		elif self.normalize == self.N_LOG_SOFTMAX:
			heuristic = {k: h.log_softmax(1) for k, h in heuristic.items()}
	
		for side in batch.keys():
			heuristic[f'{side}_mask'] = padding_mask[side]
			
		return heuristic
	