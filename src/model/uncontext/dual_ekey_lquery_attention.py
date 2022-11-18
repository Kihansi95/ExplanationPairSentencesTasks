import torch
from torch import nn

from model.layers.attention import Attention
from model.layers.fully_connected import FullyConnected
from modules.logger import log


class DualEkeyLquery(nn.Module):
	
	def __init__(self, d_embedding: int,
	             padding_idx: int,
	             vocab_size:int=None,
	             pretrained_embedding=None,
	             n_class=3,
	             concat_context=True,
	             **kwargs):
		"""
		Delta model has a customized attention layers
		"""
		
		super(DualEkeyLquery, self).__init__()
		# Get model parameters
		assert not(vocab_size is None and pretrained_embedding is None), 'Provide either vocab size or pretrained embedding'
		
		# embedding layers
		freeze = kwargs.get('freeze', False)
		
		self.n_classes = n_class
		dropout = kwargs.get('dropout', 0.)
		
		num_heads = kwargs.get('num_heads', 1)
		activation = kwargs.get('activation', 'relu')
		
		if pretrained_embedding is None:
			log.debug(f'Construct embedding from zero')
			self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embedding, padding_idx=padding_idx)
		else:
			log.debug(f'Load vector from pretraining')
			self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze, padding_idx=padding_idx)
		
		# contextual block
		d_in_contextualize = d_embedding
		d_hidden_lstm = kwargs.get('d_hidden_lstm', d_in_contextualize)
		n_context = kwargs.get('n_context', 1)
		self.bidirectional = True
		self.lstm = nn.LSTM(input_size=d_in_contextualize, hidden_size=d_hidden_lstm, num_layers=n_context, batch_first=True, bidirectional=self.bidirectional, dropout=(n_context > 1) * dropout)
		
		# Bidirectional
		d_out_contextualize = d_in_contextualize * 2
		d_attn = kwargs.get('d_attn', d_embedding) # d_attn just to project key value
		
		# Squeeze the context into dimension of key == embedding
		#self.fc_context = FullyConnected(d_out_contextualize, d_attn, activation=activation, dropout=dropout)
		
		attention_raw = kwargs.get('attention_raw', False)
		self.attention = Attention(embed_dim=d_out_contextualize, num_heads=num_heads, dropout=dropout, kdim=d_attn, vdim=d_attn, batch_first=True, attention_raw=attention_raw)
		
		d_context = d_out_contextualize
		
		d_fc_out = kwargs.get('d_fc_out', d_context)
		
		d_in_squeeze = (d_context + concat_context * d_out_contextualize) * 2
		self.concat_context = concat_context
		
		self.fc_squeeze = FullyConnected(d_in_squeeze, d_fc_out, activation=activation, dropout=dropout)
		
		n_fc_out = kwargs.get('n_fc_out', 0)
		self.fc_out = nn.ModuleList([
			FullyConnected(d_fc_out, d_fc_out, activation=activation, dropout=dropout) for _ in range(n_fc_out)
		])
		
		self.classifier = nn.Sequential(
			nn.Linear(d_fc_out, self.n_classes),
			nn.Dropout(p=dropout)
		)
		
		self.softmax = nn.Softmax(dim=1)
		
		# Constants
		self.d = 1 + int(self.lstm.bidirectional)
		
	def forward_lstm(self, x: torch.LongTensor):
		"""
        Contextualize each branch by lstm

        Args:
            x: embedding

        Returns:
        """
		x = self.embedding(x)
		n_direction = int(self.bidirectional) + 1
		
		# hidden.size() == (1, N, d_hidden_lstm)
		# hseq.size() == (N, L, d_hidden_lstm)
		self.lstm.flatten_parameters()  # flatten parameters for data parallel
		hseq, (hidden, _) = self.lstm(x)
		
		hidden = hidden[-n_direction:].permute(1, 0, 2)  # size() == (N, n_direction, d_hidden_lstm)
		hidden = hidden.reshape(hidden.size(0), 1, -1)  # size() == (N, 1, n_direction * d_hidden_lstm)
		
		return hidden, hseq
	
	def forward(self, premise_ids, hypothesis_ids, premise_padding=None, hypothesis_padding=None):
		
		# N = batch_size
		# L = sequence_length
		# h = hidden_dim = embedding_size
		# C = n_class
		ids = {'premise': premise_ids, 'hypothesis': hypothesis_ids}
		padding_mask = {'premise': premise_padding, 'hypothesis': hypothesis_padding}
		
		# Reproduce hidden representation from LSTM
		h_context = dict()
		e = dict()
		for side, x in ids.items():
			e[side] = self.embedding(x)
			
			self.lstm.flatten_parameters()  # flatten parameters for data parallel
			_, (h_last, _) = self.lstm(e[side]) # (n_direction, L, d_out_lstm)
			
			n_direction = int(self.bidirectional) + 1
			h_last = h_last[-n_direction:].permute(1, 0, 2)  # size() == (N, n_direction, d_out_lstm)
			
			# Reswapping dimension for multihead attention
			h_context[side] = h_last.reshape(h_last.size(0), 1, -1)  # (N, 1, n_direction * d_out_lstm)
	
	
		# Compute attention
		attn_weights = dict()
		context = dict()
		sides = list(ids.keys())
		for s, s_bar in sides, sides[::-1]:
			# context.size() == (N, 1, d_attention)
			# attn_weight.size() == (N, 1, L)
			context[s_bar], attn_weights[s] = self.attention(query=h_context[s_bar], key=e[s], value=e[s],key_padding_mask=padding_mask[s])
			attn_weights[s] = attn_weights[s].squeeze(1)    # attn_weights.size() == (N, L)
		
		for s in sides:
			context[s] = context[s].squeeze(dim=1)  # (N, d_attention)
			h_context[s] = h_context[s].squeeze(dim=1)      # (N, d_attention)
		
		if self.concat_context:
			x = torch.cat(list(context.values()) + list(h_context.values()), dim=1)  # (N, d_concat)
		else:
			x = torch.cat(list(context.values()), dim=1)
		x = self.fc_squeeze(x)  # (N, d_fc_out)
		
		for fc in self.fc_out:
			x = fc(x)  # (N, d_fc_out) unchanged
		out = self.classifier(x)  # (N, n_class)
		
		return out, attn_weights


if __name__ == '__main__':
	
	from collections import Counter
	import spacy
	from torch import optim
	from torch.nn.utils.rnn import pad_sequence
	from torchtext.vocab import Vocab
	from torchtext.data import get_tokenizer
	
	# === Params ===
	spacy_model = spacy.load('fr_core_news_sm')
	method = 'general'
	h = spacy_model.vocab.vectors.shape[-1]
	
	# === Examples ===
	doc1 = [
		'Bonjour tonton',
		'Comment allez-vous?',
		'Nik a les cheveux courts.'
	]
	doc2 = [
		'On l’utilise principalement entre copains, entre écoliers, entre jeunes…',
		'Ce repas/plat était très bon!',
		'Tina a les cheveux bouclés.'
	]
	y = [0, 1, 2]
	
	# Tokenize
	# ==============
	# tokenizer = Tokenizer(spacy_model=spacy_model, mode=2)
	tokenizer = get_tokenizer('spacy')  # spacy tokenizer, provided by torchtext
	counter = Counter(tokenizer(doc1 + doc2))
	
	vocab = Vocab(counter, specials=['<unk>', '<pad>', '<msk>'])
	tokenizer.vocab = vocab
	# === Test ===
	
	# tokenize
	x1 = [tokenizer.numericalize(d) for d in doc1]
	x2 = [tokenizer.numericalize(d) for d in doc2]
	
	# convert to tensor
	x1 = [torch.tensor(x, dtype=torch.long) for x in x1]
	x2 = [torch.tensor(x, dtype=torch.long) for x in x2]
	
	x1 = pad_sequence(x1, batch_first=True)
	x2 = pad_sequence(x2, batch_first=True)
	
	y = torch.tensor(y, dtype=torch.long)
	
	model = DualEkeyLquery(d_in=300, dropout=0, d_fc_lstm=-1, d_fc_attentiion=-1, d_context=-1, n_class=3, n_fc_out=0)
	
	model.train()
	
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	
	for epoch in range(1):
		# reset optimizer
		optimizer.zero_grad()
		
		preds, _ = model([x1, x2])
		loss = loss_fn(preds, y)
		
		loss.backward()
		optimizer.step()
		
		running_loss = loss.item()
		print("[{:0>3d}] loss: {:.3f}".format(epoch + 1, running_loss))
	
	model.eval()
	predict, _ = model([x1, x2])
	predict = predict.detach()
	print('Prediction:')
	print(predict)