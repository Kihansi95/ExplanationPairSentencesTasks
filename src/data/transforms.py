from typing import Union

import spacy
from torch.nn import Module

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
		rationale = rationale / rationale.sum(axis=1).unsqueeze(1)
		return entropy(rationale, padding_mask)
	
	def __str__(self):
		return 'entropy_transform'