from typing import Union, Optional, Any

import spacy
import torch
from torch import Tensor
from torch.nn import Module

from data import functional as F
from modules.const import Normalization
from modules.metrics import entropy


class SpacyTokenizerTransform(Module):
	"""
	Turns texts into Spacy tokens
	"""

	def __init__(self, spacy_model):
		super(SpacyTokenizerTransform, self).__init__()
		self.sm = spacy.load(spacy_model) if isinstance(spacy_model, str) else spacy_model
		
	def forward(self, texts):
		
		if isinstance(texts, str):
			return [tk.text for tk in self.sm(texts.strip())]
		
		docs = self.sm.pipe(texts)
		return [[tk.text for tk in doc] for doc in docs]
	
	def __str__(self):
		return 'spacy'


class LemmaLowerTokenizerTransform(SpacyTokenizerTransform):
	"""
	Transforms list of sentence into list of word array. Words are lemmatized and lower cased
	"""
	def forward(self, texts: Union[str, list]):
		
		if isinstance(texts, str):
			return [tk.lemma_.lower() for tk in self.sm(texts.strip())]
		
		texts = [t.strip() for t in texts]
		return [[tk.lemma_.lower() for tk in doc] for doc in self.sm.pipe(texts)]
	
	def __str__(self):
		return 'lemma-lower'


class PaddingToken(Module):
	
	def __init__(self, pad_value):
		super(PaddingToken, self).__init__()
		self.pad_value = pad_value
	
	def forward(self, text):
		txt_lens = [len(t) for t in text]
		max_len = max(txt_lens)
		txt_lens = [max_len - l for l in txt_lens]
		for i in range(len(text)):
			text[i] += [self.pad_value] * txt_lens[i]
		return text
	
	
class EntropyTransform(Module):
	
	def __init__(self, ):
		super(EntropyTransform, self).__init__()
	
	def forward(self, rationale, padding_mask):
		# transform into uniform distribution:
		sum_rationale = rationale.sum(axis=1).unsqueeze(1)
		rationale = rationale / (sum_rationale + (sum_rationale == 0))
		entr = entropy(rationale, padding_mask)
		return entr
	
	def __str__(self):
		return 'entropy_transform'
	
	
class NormalizationTransform(Module):
	
	def __init__(self, normalize: Normalization='std'):
		super(NormalizationTransform, self).__init__()
		self.normalize = normalize
		
	def forward(self, batch: Tensor):
		
		if self.normalize == Normalization.SOFTMAX:
			return batch.softmax(-1)
		
		if self.normalize == Normalization.LOG_SOFTMAX:
			return batch.log_softmax(-1)
		
		if self.normalize == Normalization.STANDARD:
			return batch / torch.threshold(batch.sum(-1, keepdim=True), 1, 1)
		
		if self.normalize == Normalization.LOG_STANDARD:
			_batch = batch / torch.threshold(batch.sum(-1, keepdim=True), 0, 1)
			return _batch.log()
		
		return batch


class ToTensor(Module):
	r"""Convert input to torch tensor
    
    Note:
        This class is an adaptaion from pytorch.transforms.ToTensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """
	
	def __init__(self, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long):
		super().__init__()
		self.padding_value = padding_value
		self.dtype = dtype

	def forward(self, input: Any) -> Tensor:
		return F.to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)
