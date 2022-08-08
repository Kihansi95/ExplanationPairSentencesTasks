import string
from types import Union

import torch
from torch.nn import Module

# TODO to find a better way
class FactTransform(Module):
	"""
	Turn human highlights into boolean mask, True if it's attention
	"""
	
	def forward(self, facts: list, context: list):
		# context = list of sentence, each sentence contain tokens
		context_token_length = [len(c) for c in context]
		masks = [torch.tensor(f) * l for f, l in zip(facts, context_token_length)]
		masks = torch.cat(masks)
		
		return masks