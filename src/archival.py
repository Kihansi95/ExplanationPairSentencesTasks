import torch

from modules import log, rescale


# import enchant
# fr_dict = enchant.Dict("fr_FR")

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
_debug = False


def is_masked(sentence, block_id):
    """Tell whether a sentence should be masked or not
    
    Parameters
    ----------
    sentence : list of tokens
        list of tokens. Each tokens is a dictionary.
    block_id :
        id of block
    Returns
    -------
    mask_for_sentence : boolean
        If True, sentence should be masked
    """
    
    # delete short sentence
    if len(sentence) < MIN_TOK_PER_SENT:
        if _debug: log.debug(f'Sentence too short (Actual length = {len(sentence)} while min = {MIN_TOK_PER_SENT})')
        return True
    
    # Array of Universal POStag from sentence
    upos = [token['upos'] for token in sentence]
    
    # delete sentence with no NVA
    has_nva = any(word in upos for word in ['NOUN', 'VERB', 'ADJ'])
    if not has_nva:
        if _debug: log.debug('not having NVA')
        return True
    
    # delete sents that have high NUM proportion (number_proportion > MIN_NUM_PER_SENT)
    # numpunct_proportion = sum([token['is_punct'] for token in sentence]) / len(sentence)
    numpunct_proportion = sum([x in ['NUM', 'PUNCT'] for x in upos]) / len(upos)
    if numpunct_proportion > MAX_NUM_PER_SENT:
        if _debug: log.debug(f'Too much numpunct (Ratio = {numpunct_proportion} while max = {MAX_NUM_PER_SENT})')
        return True
    
    # delete references
    is_reference = upos[:3] in [['PUNCT'] * 3, ['PUNCT', 'NUM', 'PUNCT']]  # Get first 3 upos
    if is_reference:
        if _debug: log.debug('Is reference')
        return True
    
    # delete titles sentences
    is_title = (block_id == 0) and (upos[0] == 'NUM' or upos[-1] == 'NUM')
    if is_title:
        if _debug: log.debug('Is title')
        return True
    
    # delete wrong sentences due to OCR / sentences written in another language
    # spelling = [fr_dict.check(token['form']) for token in sentence]
    # if sum(spelling)/len(spelling) < MIN_SPELLING_CORRECT:
    #	return True
    
    return False


def sentence_masking(block: dict):
    """Tell whether corresponding sentence in block should be masked corresponding to the dataset
    
    Parameters
    ----------

    Returns
    -------
    mask_for_sentence : list of boolean
        the binary along the token_blocks, that tells corresponding sentence should be masked
    """
    return [is_masked(sentence, block['block.id']) for sentence in block['sentences']]


def aggegrate_attention(inference: dict, ranking_output: dict, aggregation='sum'):
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
    pair_mask = ranking_output['pair_mask']
    padding_mask = inference['padding_mask']
    masked_score = y_score_matrix * pair_mask
    agg_attention = dict()
    
    for side, attention in inference['attention'].items():
        dim_side = 1 if side == 'source' else 0
        
        attention = inference['attention'][side]
        
        # normalize masked_score along target/source
        normalized_score = masked_score / (masked_score.sum(dim=dim_side, keepdim=True) + 1e-30)
        
        attention = attention.mul(normalized_score.unsqueeze(-1))
        
        agg_attention[side] = attention.sum(dim=dim_side)
        
        agg_attention[side] = rescale(agg_attention[side], ~padding_mask[side])
        
        agg_attention[side] = [torch.masked_select(a, m.type(torch.bool)).tolist() for a, m in
                               zip(agg_attention[side], padding_mask[side])]
    
    return agg_attention


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
