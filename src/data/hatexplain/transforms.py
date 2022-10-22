import json
from os import path

import spacy
from spacy.tokens import Doc
from torch.nn import Module


class HeuristicTransform(Module):
	
	def __init__(self, batch_tokens=None, batch_rationale=None, cache=None, spacy_model=spacy.load('en_core_web_sm'), pos_filter = None):
		super(HeuristicTransform, self).__init__()
		self.spacy_model = spacy_model
		self.POS_FILTER = pos_filter if pos_filter is not None else ['VERB', 'NOUN', 'ADJ']
		self.cache = cache
		self.freq_path = path.join(self.cache if cache is not None else '', 'anntation_lexical_frequency.json')
		self.token_freq = self._token_frequency(batch_tokens, batch_rationale)
		
	def _token_frequency(self, batch_tokens, batch_rationale):
		
		if self.cache is not None and path.exists(self.freq_path):
			with open(self.freq_path, 'r') as f:
				token_freq = json.load(f)
			return token_freq
		
		if batch_tokens is None or batch_rationale is None:
			raise ValueError(f'The caching token frequency is not established, please feed tokens and rationales')
		
		token_freq = dict()
		
		flatten_token = [tk for sent in batch_tokens for tk in sent]
		flatten_rationale = [r for sent in batch_rationale for r in sent]
		
		for t, r in zip(flatten_token, flatten_rationale):
			if r: token_freq[t] = token_freq.get(t, 0) + 1
		
		total_freq = sum(token_freq.values())
		token_freq = {k: v / total_freq for k, v in token_freq.items()}
		token_freq = dict(sorted(token_freq.items(), key=lambda item: -item[1]))
		
		if self.cache is not None:
			with open(self.freq_path, 'r') as f:
				json.dump(f, self.freq_path)
		
		return token_freq
	
	def forward(self, batch_tokens):

		docs = [Doc(self.spacy_model.vocab, words=sent) for sent in batch_tokens]
		tokenized_docs = list(self.spacy_model.pipe(docs))
		
		pos_filter = [[tk.pos_ in ['ADJ'] for tk in d] for d in docs]
		stop_filter = [[not tk.is_stop for tk in d] for d in docs]
		mask = [pos_ and stop_ for pos_, stop_ in zip(pos_filter, stop_filter)]
		
		## Count words
		heuristics = []
		for sent_tokens, sent_mask in zip(batch_tokens, mask):
			heuris_map = [self.token_freq.get(tk, 0) for tk in sent_tokens]
			heuris_map = [h * float(m) for h, m in zip(heuris_map, sent_mask)]
			# sum_heuris = max(sum(heuris_map), 1)
			# heuris_map = [h / sum_heuris for h, m in zip(heuris_map, sent_mask)]
			heuristics.append(heuris_map)
			
		return heuristics
		