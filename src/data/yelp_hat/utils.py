import re

from modules.logger import log


def yelp_hat_ham(html):

    p = re.compile('<span(.*?)/span>')
    all_span_items = p.findall(html)
    
    if all_span_items[-1] == '><': all_span_items = all_span_items[:-1]

    return [int('class="active"' in span_item) for span_item in all_span_items]

def yelp_hat_token(html):
    p = re.compile(r'<span[^>]*>(.+?)</span>')
    tokens = p.findall(html)
    return tokens


def generate_binary_human_attention_vector(html, num_words_in_review, max_words):
	# Function provided by the dataset :
	# https://github.com/cansusen/Human-Attention-for-Text-Classification/blob/master/generate_ham/sample_generate.ipynb
	
	p = re.compile('<span(.*?)/span>')
	all_span_items = p.findall(html)
	
	if html == '{}':
		log.error('Empty human annotation - This should never print')
		return [0] * max_words
	
	if len(all_span_items) == num_words_in_review + 1:
		if (all_span_items[num_words_in_review] == '><') or (all_span_items[num_words_in_review] == ' data-vivaldi-spatnav-clickable="1"><'):
			
			binarized_human_attention = [0] * max_words
			for i in range(0, len(all_span_items) - 1):
				if 'class="active"' in all_span_items[i]:
					binarized_human_attention[i] = 1
		
		else:
			log.error('This should never print.')
	else:
		log.error('This should never print.')
	
	return binarized_human_attention