from typing import List

import pandas as pd
import torch

from data_module.fr_xnli_module import FrXNLIDM
from modules import rescale, log
from modules.utils import topk_2d


# fr_dict = enchant.Dict("fr")



def get_block(df_block, uid, feature='norm'):
	"""Get features from block given its uid

	Parameters
	----------
	df_block : pandas.DataFrame
		database of block
	uid : str
		block unique id indexed in block df
	feature : str
		if feature is among `['text', 'tokens']`, returns the equivalent features from dataframe.
		feature should be among token features (`norm`, `form`, ...) to generate appropriate block.

	Returns
	-------
		list of list
		tokens or text from text block

	Examples
	--------
	```
	uid_source = 'FMSH_PB188a_00043_004_04'
	token_block_source = get_block(df_block, uid_source, 'norm')
	```
	"""
	
	if feature in ['text', 'token']:
		return df_block.loc[uid, feature]
	
	block = df_block.loc[uid, 'sents']
	block = [[tk[feature] for tk in sent] for sent in block]
	return block


MIN_TOK_PER_SENT = 5
MIN_SPELLING_CORRECT = 0.6
MAX_NUM_PER_SENT = 0.4


def is_masked(sentence, block_id):
	"""Tell whether a sentence should be masked or not

	Parameters
	----------
	sentence : #TODO
	block_id :

	Returns
	-------

	"""
	
	# delete short sentence
	if len(sentence) < MIN_TOK_PER_SENT:
		return True
	
	# delete sentence with no NVA
	has_nva = any([token['is_nva'] for token in sentence])
	if not has_nva:
		return True
	
	# delete sents that have high NUM proportion (number_proportion > MIN_NUM_PER_SENT)
	numpunct_proportion = sum([token['is_punct'] for token in sentence]) / len(sentence)
	if numpunct_proportion > MAX_NUM_PER_SENT:
		return True
	
	# delete references
	first_3_upos = [token['upos'] for token in sentence[:3]]
	is_reference = first_3_upos in [['PUNCT'] * 3, ['PUNCT', 'NUM', 'PUNCT']]
	if is_reference:
		return True
	
	# delete titles sentences
	is_title = (block_id == 0) and (sentence[0]['upos'] == 'NUM' or sentence[-1]['upos'] == 'NUM')
	if is_title:
		return True
	
	# delete wrong sentences due to OCR / sentences written in another language
	# spelling = [fr_dict.check(token['form']) for token in sentence]
	# if sum(spelling)/len(spelling) < MIN_SPELLING_CORRECT:
	# 	return True
	
	return False


def sentence_masking(token_blocks: list):
	"""Tell whether corresponding sentence in block should be masked corresponding to the dataset

	Parameters
	----------
	token_blocks : list of list
		list of token sentence. Token sentence = a list of tokens. Each tokens is a dictionary.

	Returns
	-------
	mask_for_sentence : list of boolean
		the binary along the token_blocks, that tells corresponding sentence should be masked
	"""
	
	return [is_masked(sentence) for sentence in token_blocks]


def aggegrate_attention(inference: dict,
                        aggregation='threshold',
                        **kwargs):
	"""Aggregate attention along target/source

	Parameters
	----------
	inference :
	ranking_output :
	aggregation : average, threshold, top_k, maximum

	Returns
	-------

	"""
	
	y_score_matrix = inference['y_score_matrix']
	
	padding_mask = inference['padding_mask']
	
	masked_score = y_score_matrix.clone()
	
	if aggregation == 'threshold':
		assert 'epsilon' in kwargs, 'epsilon_score should be provided for threshold aggregation'
		assert 0. <= kwargs['epsilon'] <= 1., 'epsilon should be between 0 and 1'
		epsilon = kwargs['epsilon']
		pair_mask = (y_score_matrix >= epsilon).type(torch.float)
		masked_score *= pair_mask  # We ignore the attention maps from the low scoring pairs
	
	if aggregation == 'topk':
		assert 'k' in kwargs, 'k should be provided for topk aggregation'
		assert 0 < kwargs['k'], 'k should be positive'
		k = min(kwargs['k'], y_score_matrix.numel()) # if k larger than number of pairs, take all pairs
		top_k = y_score_matrix.reshape(-1).topk(k=k, largest=True)
		kth_value = top_k.values[-1]
		pair_mask = (y_score_matrix >= kth_value).type(torch.float)
		masked_score *= pair_mask  # keep only attention maps from k best scoring pairs
	
	if aggregation == 'average':
		masked_score = torch.ones_like(y_score_matrix)
		
	if aggregation == 'topk_threshold':
		assert 'epsilon' in kwargs, 'epsilon_score should be provided for threshold aggregation'
		assert 0. <= kwargs['epsilon'] <= 1., 'epsilon should be between 0 and 1'
		assert 'k' in kwargs, 'k should be provided for topk aggregation'
		assert 0 < kwargs['k'], 'k should be positive'
		
		# get the top k highest scores
		k = min(kwargs['k'], y_score_matrix.numel())  # if k larger than number of pairs, take all pairs
		top_k_values, top_k_indices = y_score_matrix.view(-1).topk(k)
		
		pair_mask = torch.zeros_like(y_score_matrix)
		rows = top_k_indices.div(y_score_matrix.size(1), rounding_mode='trunc')
		cols = top_k_indices % y_score_matrix.size(1)
		pair_mask[rows, cols] = 1.

		# get those above epsilon
		epsilon = kwargs['epsilon']
		pair_mask *= (y_score_matrix >= epsilon).type(torch.float)
		
		masked_score *= pair_mask
	
	agg_attention = dict()
	
	for side, attention in inference['attention'].items():
		dim_side = 1 if side == 'source' else 0
		
		attention = inference['attention'][side]
		
		# normalize masked_score along target/source
		normalized_score = masked_score / (masked_score.sum(dim=dim_side, keepdim=True) + 1e-30)
		
		attention = attention.mul(normalized_score.unsqueeze(-1))
		
		agg_attention[side] = attention.sum(dim=dim_side)
		
		agg_attention[side] = rescale(agg_attention[side], ~padding_mask[side])
		
		agg_attention[side] = [torch.masked_select(a, m.type(torch.bool)).tolist() for a, m in zip(agg_attention[side], padding_mask[side])]
	
	return agg_attention


def inference_block(source, target, dm, model, idx_class=1):
	"""Make prediction from source and target.

	Use datamodule to transform tokens (`norm`) into ids in embedding. The model predicts based on ids.

	Parameters
	----------
	source : list of list
		tokens from source block
	target : list
		tokens from train set
	dm : pl.DataModule
		Contain information for preprocessing inputs
	model : torch.nn.Module
		Trained model
	idx_class : int
		Index of class used for inference. Default is 1 (entailment).

	Returns
	-------
	y_score_matrix : 2D float matrix (list of list)
		matrix of probability of entailment from model

	attention : 3D float matrix
		2D matrix of attention.
		Matrix indexing:
			attentions[idx_sent_source][idx_sent_target][side].shape() == sentence_length from side

	padding_mask : 3D float matrix
		2D matrix of padding mask vector of L. Same dimension as `attention`

	Examples
	--------
	score = inference_block(source, target, dm, model)
	# get attention of the source sentence in

	"""
	sent_pairs = {'source': [], 'target': []}
	align_idx = {'source': [], 'target': []}
	
	for idx_src, sent_src in enumerate(source):
		for idx_target, sent_target in enumerate(target):
			sent_pairs['source'].append(sent_src)
			sent_pairs['target'].append(sent_target)
			
			# re align for output score
			align_idx['source'].append(idx_src)
			align_idx['target'].append(idx_target)
	
	# preprocess data using datamodule
	if isinstance(dm, FrXNLIDM):
		batch = dm.collate({
			'premise.tokens': sent_pairs['source'],
			'hypothesis.tokens': sent_pairs['target'],
		})
	else:
		batch = dm.collate({
			'premise.norm': sent_pairs['source'],
			'hypothesis.norm': sent_pairs['target'],
		})
	
	
	with torch.no_grad():
		device = model.device
		y_hat, a_hat = model(premise_ids=batch['premise.ids'].to(device),
		                     hypothesis_ids=batch['hypothesis.ids'].to(device),
		                     premise_padding=batch['padding_mask']['premise'].to(device),
		                     hypothesis_padding=batch['padding_mask']['hypothesis'].to(device))
	
	# normalize output
	y_hat = y_hat.softmax(-1)
	y_hat = y_hat[:, idx_class]
	a_hat = {s: a_hat[s].softmax(-1) for s in a_hat}
	
	N_SOURCE = len(source)
	N_TARGET = len(target)
	L_SOURCE = a_hat['premise'].size(-1)
	L_TARGET = a_hat['hypothesis'].size(-1)
	
	# construct matrix
	y_score_matrix = torch.ones([N_SOURCE, N_TARGET]) * -1
	a_hat_matrix = {
		'source': torch.zeros([N_SOURCE, N_TARGET, L_SOURCE]),
		'target': torch.zeros([N_SOURCE, N_TARGET, L_TARGET])
	}
	
	padding_mask = {
		'source': torch.zeros([N_SOURCE, L_SOURCE], dtype=torch.bool),
		'target': torch.zeros([N_TARGET, L_TARGET], dtype=torch.bool),
	}
	
	for i in range(len(y_hat)):
		index_source = align_idx['source'][i]
		index_target = align_idx['target'][i]
		
		y_score_matrix[index_source, index_target] = y_hat[i]
		a_hat_matrix['source'][index_source, index_target] = a_hat['premise'][i]
		a_hat_matrix['target'][index_source, index_target] = a_hat['hypothesis'][i]
		
		padding_mask['source'][index_source] = ~batch['padding_mask']['premise'][i]
		padding_mask['target'][index_target] = ~batch['padding_mask']['hypothesis'][i]
	
	return {
		'y_score_matrix': y_score_matrix,
		'attention': a_hat_matrix,
		'padding_mask': padding_mask,
	}


def ranking_score(inference, epsilon_covery=0.5, aggregation='development', **kwargs):
	"""
	epsilon_score: float, default 0.8
		Value at which, precision(y_hat >= epsilon_score) ~ 1
	epsilon_covery: float, default 0.5
		Because we would look into column and row if the detection is very sparse
	"""
	y_score_matrix = inference['y_score_matrix']
	
	# We define a covery of positive pair
	if aggregation == 'development':

		epsilon = kwargs['epsilon']
		pair_mask = (y_score_matrix >= epsilon).type(torch.float)

		if (y_score_matrix.size(0) > 1 or y_score_matrix.size(1) > 1) and pair_mask.mean() < epsilon_covery:
			# Detect relationship as row or column if they have more than 1 row and more than 1 column
			inference_col = torch.prod(pair_mask, 0)
			inference_row = torch.prod(pair_mask, 1)

			if inference_col.any().item():
				relation_type = 'column'
				masking_score = y_score_matrix.mul(inference_col.unsqueeze(0))
				ranking_score = masking_score[masking_score > 0].mean()

			if inference_row.any().item():
				if relation_type is not None:
					relation_type += '_row'
				else:
					relation_type = 'row'

				masking_score = y_score_matrix.mul(inference_row.unsqueeze(1))
				ranking_score = masking_score[masking_score > 0].mean()

			if relation_type is None:
				ranking_score = y_score_matrix.mul(y_score_matrix > epsilon).mean()
				relation_type = 'neutral'

		else:
			# Compute the average score.
			# Later: reweight the score with number of sentence
			# Later: reweight (or not) the score with sentence length (see if the quality depend on sentence length)
			ranking_score = y_score_matrix.mul(pair_mask).mean()
			relation_type = 'dense' if ranking_score > 0.8 else 'neutral'

		return {
			'ranking_score': ranking_score.item(),
			'type': relation_type,
			'pair_mask': pair_mask,
		}

	elif aggregation == 'topk_threshold':

		assert 'epsilon' in kwargs, 'epsilon_score should be provided for threshold aggregation'
		assert 0. <= kwargs['epsilon'] <= 1., 'epsilon should be between 0 and 1'
		assert 'k' in kwargs, 'k should be provided for topk aggregation'
		assert 0 < kwargs['k'], 'k should be positive'

		epsilon = kwargs['epsilon']

		# filter all the score below a score
		mask_threshold = y_score_matrix >= epsilon
		y_score_matrix = y_score_matrix * mask_threshold

		# if every one die, no inference
		if y_score_matrix.sum() == 0.:
			return {
				'relation_type': 'neutral',
				'ranking_score': 0.,
				'pair_mask': mask_threshold,
			}

		# get the top k highest scores
		k = min(kwargs['k'], y_score_matrix.numel())  # if k larger than number of pairs, take all pairs
		
		#top_k = y_score_matrix.reshape(-1).topk(k=k, largest=True)
		#kth_value = top_k.values[-1]
		#mask_topk = y_score_matrix >= kth_value
		
		top_k_values, top_k_indices = torch.topk(y_score_matrix.view(-1), k)
		
		rows = top_k_indices.div(y_score_matrix.size(1), rounding_mode='trunc')
		cols = top_k_indices % y_score_matrix.size(1)
		
		# Mark these indices in the mask
		mask_topk = torch.zeros_like(y_score_matrix, dtype=torch.bool)
		mask_topk[rows, cols] = True
		
		ranking_score = y_score_matrix[mask_topk].mean()
		

		return {
			'ranking_score': ranking_score.item(),
			'type': 'inference',
			'pair_mask': mask_topk,
		}

	else :
		raise NotImplementedError()


def extract_string_id(block_alto_id, block_attention):
	"""
	get attention value for corresponding string alto ids

	block_alto_id : list of list
		list of ids sentence. Ids sentence = a list of alto ids

	block_attention : list of list
		list of attention sentence. Attention sentence = a list of attention along s
	"""
	
	assert len(block_alto_id) == len(
		block_attention), f'Incompatible number of sentence. len(block_alto_id)={len(block_alto_id)}, len(block_attention)={len(block_attention)}'
	
	attention_values = [
		{
			'alto_id': alto_id,
			'attention': a_hat,
		} for sentence_alto_id, sentence_attention in zip(block_alto_id, block_attention)
		for alto_id, a_hat in zip(sentence_alto_id, sentence_attention) if a_hat > 0
	]
	
	return attention_values


def get_top_sentence_pairs(df_block: pd.DataFrame, uid_source: str, uid_target: str, inference, k: int = -1, above_threshold: float = 0.8):
	"""Get the scored sentence pairs of a block
	
	Parameters
	----------
	df_block : pd.DataFrame
		text blocks in database
	uid_source : str
		identity of source block
	uid_target : str
		identity of target block
	inference : dict
		dictionary containing inference information
	k : int, Default: -1
		number of top highest predicted pairs to get.
		If -1, get all pairs.
	above_threshold : float, Default: 0.8
		threshold to filter pairs. Only pairs with score above threshold are returned.

	Returns
	-------
	top_sentence_pairs: list
		list of sentence pairs. Each element is a dictionary of following keys :
		`idx_source` : int
		    idx of sentence in source block
		`idx_target` : int
			idx of sentence in target block
		`y_score` : float
			prediction score that the sentence is an inference
		`pair_text` : str
			concatenation of the two sentences text
		`pair_tokens` : list of token
			1d array containing tokens of the pair
		`pair_attention` : list of token
			1d array containing attention value of the pair
	"""
	source_block_form = get_block(df_block, uid_source, 'form')
	target_block_form = get_block(df_block, uid_target, 'form')

	source_block_space = get_block(df_block, uid_source, 'space_after')
	target_block_space = get_block(df_block, uid_target, 'space_after')

	source_block_form = [[form + (' ' if space_after else '') for form, space_after in zip(source_sent_form, source_sent_space)] for source_sent_form, source_sent_space in zip(source_block_form, source_block_space)]
	target_block_form = [[form + (' ' if space_after else '') for form, space_after in zip(target_sent_form, target_sent_space)] for target_sent_form, target_sent_space in zip(target_block_form, target_block_space)]

	y_score_matrix = inference['y_score_matrix']
	
	assert len(source_block_form) == y_score_matrix.size(0), f'Incompatible number of sentence. len(block_source)={len(source_block_form)}, y_score_matrix.size(0)={y_score_matrix.size(0)}'
	assert len(target_block_form) == y_score_matrix.size(1), f'Incompatible number of sentence. len(block_target)={len(target_block_form)}, y_score_matrix.size(1)={y_score_matrix.size(1)}'
	
	top_values, (idx_sent_source, idx_sent_target) = topk_2d(y_score_matrix, k=k)
	if k > 0:
		top_values = top_values[top_values > above_threshold]
		idx_sent_source = idx_sent_source[:len(top_values)]
		idx_sent_target = idx_sent_target[:len(top_values)]
	
	source_block_text = [
		''.join([form + (' ' if space_after else '') for form, space_after in zip(sent_form, sent_space)]) for sent_form, sent_space in zip(source_block_form, source_block_space)
	]

	target_block_text = [
		''.join([form + (' ' if space_after else '') for form, space_after in zip(sent_form, sent_space)])
		for sent_form, sent_space in zip(target_block_form, target_block_space)
	]

	attention = inference['attention']
	padding_mask = inference['padding_mask']	
	
	return [
		{
			'idx_sent_source': idx_source.item(),
			'idx_sent_target': idx_target.item(),
			'y_score': y_score.item(),
			'attention_source': attention['source'][idx_source, idx_target][padding_mask['source'][idx_source]].tolist(),
			'attention_target': attention['target'][idx_source, idx_target][padding_mask['target'][idx_target]].tolist(),
		    'pair_text': source_block_text[idx_source] + ' ' + target_block_text[idx_target],
			'pair_token' : source_block_form[idx_source] + target_block_form[idx_target],
			'pair_attention' : attention['source'][idx_source, idx_target][padding_mask['source'][idx_source]].tolist() + attention['target'][idx_source, idx_target][padding_mask['target'][idx_target]].tolist(),
		} for y_score, idx_source, idx_target in zip(top_values, idx_sent_source, idx_sent_target)
	]


