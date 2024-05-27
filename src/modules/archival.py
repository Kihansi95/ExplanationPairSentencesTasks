from itertools import product

import torch

from data_module.fr_xnli_module import FrXNLIDM
from modules import rescale


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
    
    block = df_block.loc[uid, 'sentences']
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
    
    y_score_matrix = inference['sentences.y_score']
    
    padding_mask = inference['sentences.padding_mask']
    
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
        k = min(kwargs['k'], y_score_matrix.numel())  # if k larger than number of pairs, take all pairs
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
        # print('masked_score*pairmask', masked_score)
    
    agg_attention = dict()
    
    for side, attention in inference['sentences.attention'].items():
        dim_side = 1 if side == 'source' else 0
        
        attention = inference['sentences.attention'][side]
        
        # normalize masked_score along target/source
        normalized_score = masked_score / (masked_score.sum(dim=dim_side, keepdim=True) + 1e-30)
        
        attention = attention.mul(normalized_score.unsqueeze(-1))
        
        agg_attention[side] = attention.sum(dim=dim_side)
        
        agg_attention[side] = rescale(agg_attention[side], padding_mask[side])
        
        agg_attention[side] = [torch.masked_select(a, m.type(torch.bool)).tolist() for a, m in
                               zip(agg_attention[side], ~padding_mask[side])]
    
    return agg_attention


def aggregate_score(inference, aggregation='topk_threshold', **kwargs):
    """
    epsilon_score: float, default 0.8
        Value at which, precision(y_hat >= epsilon_score) ~ 1
    epsilon_covery: float, default 0.5
        Because we would look into column and row if the detection is very sparse
    """
    y_score_matrix = inference['sentences.y_score']
    
    # We define a covery of positive pair
    if aggregation == 'development':
        
        epsilon = kwargs['epsilon']
        epsilon_covery = kwargs['epsilon_covery']
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
            'blocks.score': ranking_score.item(),
            'blocks.type': relation_type,
            'blocks.pair_mask': pair_mask,
        }
    
    elif aggregation == 'topk_threshold':
        
        epsilon = kwargs.get('epsilon', None)
        k = kwargs.get('k', None)
        
        assert (epsilon is None) or (0. < epsilon < 1.), f'epsilon should be between 0 and 1. epsilon={epsilon}'
        assert (k is None) or (k > 0), f'k should be positive if given. k={k}'
        
        # Flatten the y_score_matrix
        y_score_flat = y_score_matrix.view(-1)
        
        # Order the y_score matrix by descending order
        sorted_indices = torch.argsort(y_score_flat, descending=True)
        
        # If k is specified, take only top k scores and indices
        if k is not None:
            k = min(k, y_score_matrix.numel())  # if k larger than number of pairs, take all pairs
            sorted_indices = sorted_indices[:k]
            
        # Ignore indices that are below the threshold
        sorted_indices = sorted_indices[y_score_flat[sorted_indices] >= epsilon]
        
        # Gather the corresponding y_scores
        y_scores = y_score_flat[sorted_indices]
        
        # Get the source and target indices
        S = y_score_matrix.shape[1]
        s = sorted_indices.div(S, rounding_mode='trunc') # source indices
        t = sorted_indices.remainder(S) # target indices
        
        # build mask
        # 1. mask all but the top k-values
        pair_mask = torch.zeros_like(y_score_matrix, dtype=torch.bool)
        pair_mask[s, t] = True
        
        # 2. mask those below the epsilon threshold
        pair_mask *= y_score_matrix > epsilon
        
        # aggregate score and type
        ranking_score = y_scores.mean()
        type = 'neutral' if ranking_score == 0 else 'inference'
        
        # prepare output
        inference = inference.copy()
        inference['blocks.score'] = ranking_score.item()
        inference['blocks.type'] = type
        inference['blocks.pair_mask'] = pair_mask
        inference['blocks.pair_scores'] = y_score_matrix * pair_mask
        inference['blocks.top_sentence_pairs'] = []
        for idx_source, idx_target, score in zip(s, t, y_scores):
            inference['blocks.top_sentence_pairs'].append({
                'source': idx_source.item(),
                'target': idx_target.item(),
                'y_score': score.item()
            })
        
        return inference
    
    else:
        raise NotImplementedError()

def inference_block(source, target, dm, model, idx_class=1, temperature_y=1.):
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
    temperature_y : float
        Temperature for softmax on y_hat. Default is 1.

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
        
    device = model.device
    with torch.no_grad():
        
        y_hat, a_hat = model(premise_ids=batch['premise.ids'].to(device),
                             hypothesis_ids=batch['hypothesis.ids'].to(device),
                             premise_padding=batch['padding_mask']['premise'].to(device),
                             hypothesis_padding=batch['padding_mask']['hypothesis'].to(device))
    
   
    # normalize output
    y_hat = y_hat / temperature_y
    #log.debug(f'y_hat ({y_hat.shape}) : {y_hat}')
    y_hat = y_hat.softmax(1)
    #log.debug(f'y_hat.softmax : {y_hat}')
    #sum_y_hat = y_hat.sum(1)
    #log.debug(f'sum_y_hat : {sum_y_hat}')
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


def inference_block_with_mask(source, target, dm, model, idx_class=1, temperature_y=1.):
    """New fucntion of inference block. Function take into account of block structure.

    Parameters
    ----------
    source : dict
        one row from df_block
    target : dict
        one row from df_block
    dm : pl.DataModule
        Contain information for preprocessing inputs
    model : torch.nn.Module
        Trained model
    idx_class : int
        Index of class used for inference. Default is 1 (entailment).
    temperature_y : float
        Temperature for softmax on y_hat. Default is 1.

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
    source = df_block.iloc[0]
    target = df_block.iloc[1]
    inference_score = inference_block(source, target, dm, model)
    y_score_matrix  = inference_score['y_score_matrix']
    attention_pairs = inference_score['attention']
    """
    
    source_masked = source['sentences_masked']
    target_masked = target['sentences_masked']
    S = len(source_masked)
    T = len(target_masked)
    # log.debug(f'source_masked; {source_masked}')
    # log.debug(f'target_masked; {target_masked}')
    
    sentence_pairs = [{'source': s_sent, 'target': t_sent} for s_sent, t_sent in
                      product(source['sentences'], target['sentences'])]
    source_norms = [[token['norm'] for token in pair['source']] for pair in sentence_pairs]
    target_norms = [[token['norm'] for token in pair['target']] for pair in sentence_pairs]
    sentence_pairs = {'source': source_norms, 'target': target_norms}
    
    masked_pairs = [s_masked or t_masked for s_masked in source_masked for t_masked in target_masked]
    
    # preprocess data using datamodule
    if isinstance(dm, FrXNLIDM):
        batch = dm.collate({
            'premise.tokens': sentence_pairs['source'],
            'hypothesis.tokens': sentence_pairs['target'],
        })
    else:
        batch = dm.collate({
            'premise.norm': sentence_pairs['source'],
            'hypothesis.norm': sentence_pairs['target'],
        })
        
    device = model.device
    with torch.no_grad():
        y_hat, a_hat = model(premise_ids=batch['premise.ids'].to(device),
                             hypothesis_ids=batch['hypothesis.ids'].to(device),
                             premise_padding=batch['padding_mask']['premise'].to(device),
                             hypothesis_padding=batch['padding_mask']['hypothesis'].to(device))
    
    # normalize output
    y_hat = y_hat / temperature_y
    y_hat = y_hat.softmax(-1)
    y_hat = y_hat[:, idx_class]
    a_hat = {s: a_hat[s].softmax(-1) for s in a_hat}
    
    # mask out scores
    y_mask = torch.tensor(masked_pairs, dtype=torch.bool, device=device)
    
    y_hat = y_hat * ~y_mask  # mask out scores
    y_score_matrix = y_hat.reshape(S, T)  # reshape to matrix
    y_score_matrix = y_score_matrix.cpu()  # move to cpu
    
    a_hat = {side : a_hat[side] * ~y_mask.unsqueeze(-1) for side in a_hat}  # mask out attention
    a_hat_matrix = {side : a_hat[side].reshape(S, T, -1) for side in a_hat}  # reshape to matrix
    a_hat_matrix = {side : a_hat_matrix[side].cpu() for side in a_hat_matrix}  # move to cpu
    
    # rename from premise/hypothesis to source/target
    a_hat_matrix = {'source': a_hat_matrix['premise'], 'target': a_hat_matrix['hypothesis']}
    padding_mask = {'source': batch['padding_mask']['premise'], 'target': batch['padding_mask']['hypothesis']}
    
    # reshape padding_mask to avoid having duplicated tensors
    padding_mask = {side : mask.reshape(S, T, -1) for side, mask in padding_mask.items()}
    
    padding_mask['source'] = padding_mask['source'][:, 0, :]
    padding_mask['target'] = padding_mask['target'][0, :, :]
    
    return {
        'sentences.y_score': y_score_matrix,
        'sentences.attention': a_hat_matrix,
        'sentences.padding_mask': padding_mask,
    }

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


# def get_top_sentence_pairs(inference, k: int = -1, above_threshold: float = 0.5):
#     """Get the scored sentence pairs of a block
#
#     Parameters
#     ----------
#     df_block : pd.DataFrame
#         text blocks in database
#     uid_source : str
#         identity of source block
#     uid_target : str
#         identity of target block
#     inference : dict
#         dictionary containing inference information
#     k : int, Default: -1
#         number of top highest predicted pairs to get.
#         If -1, get all pairs.
#     above_threshold : float, Default: 0.8
#         threshold to filter pairs. Only pairs with score above threshold are returned.
#
#     Returns
#     -------
#     top_sentence_pairs: list
#         list of sentence pairs. Each element is a dictionary of following keys :
#         `idx_source` : int
#             idx of sentence in source block
#         `idx_target` : int
#             idx of sentence in target block
#         `y_score` : float
#             prediction score that the sentence is an inference
#         `pair_text` : str
#             concatenation of the two sentences text
#         `pair_tokens` : list of token
#             1d array containing tokens of the pair
#         `pair_attention` : list of token
#             1d array containing attention value of the pair
#     """
#     y_score_matrix = inference['y_score_matrix']
#     attention = inference['attention']
#     # padding_mask = inference['padding_mask']
#
#     # Flatten the y_score_matrix
#     y_score_flat = y_score_matrix.view(-1)
#
#     # Order the y_score matrix by descending order
#     sorted_indices = torch.argsort(y_score_flat, descending=True)
#
#     # If k is specified, take only top k scores and indices
#     if k is not None:
#         sorted_indices = sorted_indices[:k]
#
#     # Ignore indices that are below the threshold
#     sorted_indices = sorted_indices[y_score_flat[sorted_indices] >= above_threshold]
#
#     # Gather the corresponding y_scores
#     y_scores = y_score_flat[sorted_indices]
#
#     # Extract corresponding attention
#     attention_source = attention['source'].view(-1, attention['source'].shape[-1])[sorted_indices]
#     attention_target = attention['target'].view(-1, attention['target'].shape[-1])[sorted_indices]
#
#     # log.debug(f'padding.source.flatten = {mask_source}')
#
#     # Get the source and target indices
#     n_col = y_score_matrix.shape[1]
#     source_indices = sorted_indices.div(n_col, rounding_mode='trunc')
#     target_indices = sorted_indices.remainder(n_col)
#
#     # log.debug(f'attention.target.i = {attention_target[1]}')
#     # log.debug(f'mask.target.i = {~mask_target[1]}')
#     # log.debug(f'attention.masked.target.i = {attention_target[1][~mask_target[1]]}')
#
#     # Construct the list of sentence pairs
#     sentence_pairs = [{
#         'idx_sent_source': source_indices[i].item(),
#         'idx_sent_target': target_indices[i].item(),
#         'y_score': y_scores[i].item(),
#         # 'attention_source': attention_source[i][~mask_source[i]],
#         # 'attention_target': attention_target[i][~mask_target[i]],
#         'attention_source': attention_source[i],
#         'attention_target': attention_target[i],
#     } for i in range(len(sorted_indices))]
#
#     return sentence_pairs

def aggregate_sentence_pairs(inference,
                             inference_pairs,
                             aggregation_score='mean',
                             aggregate_attention='weighted_aggregate'):
    
    ranking_score = -1
    attention = None
    
    if aggregation_score == 'mean':
        top_scores = [p['y_score'] for p in inference_pairs]
        ranking_score = sum(top_scores) / len(top_scores)
    
    if aggregate_attention == 'weighted_aggregate':
        
        y_score_matrix = inference['y_score_matrix']
        padding_mask = inference['padding_mask']
        
        masked_score = torch.zeros_like(y_score_matrix)
        for p in inference_pairs:
            masked_score[p['idx_sent_source'], p['idx_sent_target']] = 1.
        masked_score *= y_score_matrix
        
        agg_attention = dict()
        
        for side, attention in inference['attention'].items():
            dim_side = 1 if side == 'source' else 0
            
            attention = inference['attention'][side]
            
            # normalize masked_score along target/source
            normalized_score = masked_score / (masked_score.sum(dim=dim_side, keepdim=True) + 1e-30)
            
            attention = attention.mul(normalized_score.unsqueeze(-1))
            
            agg_attention[side] = attention.sum(dim=dim_side)
            
            agg_attention[side] = rescale(agg_attention[side], padding_mask[side])
            
            agg_attention[side] = [torch.masked_select(a, ~m.type(torch.bool)).tolist() for a, m in zip(agg_attention[side], padding_mask[side])]
            
    
    return {
        'score': ranking_score,
        'attention.source': agg_attention['source'],
        'attention.target': agg_attention['target'],
    }

def aggregate_attention_new(inference,
                            aggregation='weighted_average',
                            **kwargs):
    
    inference = inference.copy()
    padding_mask = inference['sentences.padding_mask']
    
    if aggregation == 'weighted_average':
        block_attention = dict()
        
        for side, attention in inference['sentences.attention'].items():
            dim_side = 1 if side == 'source' else 0
            
            attention = inference['sentences.attention'][side]
            mask = padding_mask[side]
            
            # normalize masked_score along target/source
            masked_score = inference['blocks.pair_scores']
            
            normalized_score = masked_score / (masked_score.sum(dim=dim_side, keepdim=True) + 1e-30)
            
            # reweigh attention weights by the predictive score
            attention = attention.mul(normalized_score.unsqueeze(-1))
            
            # sum the same sentence across different pairs
            attention = attention.sum(dim=dim_side)
            
            # rescale to get between 1 and 0
            attention = rescale(attention, mask)
            
            block_attention[side] = [torch.masked_select(a, ~m).tolist() for a, m in zip(attention, mask)]
    
        inference['blocks.attention'] = block_attention
        return inference
        
    elif aggregation == 'and':
        block_attention = dict()
        
        for side, attention in inference['sentences.attention'].items():
            dim_side = 1 if side == 'source' else 0
            
            attention = inference['sentences.attention'][side]
            mask = padding_mask[side]
            
            # sum the same sentence across different pairs
            attention = attention.mul(dim=dim_side)
            
            # rescale to get between 1 and 0
            attention = rescale(attention, mask)
            
            block_attention[side] = [torch.masked_select(a, ~m).tolist() for a, m in zip(attention, mask)]
        
        inference['blocks.attention'] = block_attention
        return inference
    
    else:
        raise NotImplementedError
        
    
    
    