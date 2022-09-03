from model import Net

import torch
from torch import nn

from model.layers.attention import Attention
from model.layers.fully_connected import FullyConnected
from modules.logger import log


class PairLstmAttention(Net):
	
	def __init__(self, d_embedding: int, padding_idx: int, vocab_size:int=None, pretrained_embedding=None, n_class=3, **kwargs):
		"""
		Delta model has a customized attention layers
		"""
		
		super(PairLstmAttention, self).__init__()
		# Get model parameters
		assert not(vocab_size is None and pretrained_embedding is None), 'Provide either vocab size or pretrained embedding'
		
		# embedding layers
		freeze = kwargs.get('freeze', False)
		
		self.n_classes = n_class
		dropout = kwargs.get('dropout', 0.)
		self.bidirectional = True  # force to default
		
		num_heads = kwargs.get('num_heads', 1)
		activation = kwargs.get('activation', 'relu')
		
		if pretrained_embedding is None:
			log.debug(f'Construct embedding from zero')
			self.embedding = nn.Embedding(len(vocab), d_embedding, padding_idx=padding_idx)
		else:
			log.debug(f'Load vector from pretraining')
			self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze, padding_idx=padding_idx)
		
		# LSTM block
		d_in_lstm = d_embedding
		d_hidden_lstm = kwargs.get('d_hidden_lstm', d_in_lstm)
		n_lstm = kwargs.get('n_lstm', 1)
		self.lstm = nn.LSTM(input_size=d_in_lstm, hidden_size=d_hidden_lstm, num_layers=n_lstm, batch_first=True,
		                    bidirectional=True, dropout=(n_lstm > 1) * dropout)
		
		# Bidirectional
		if self.bidirectional: d_hidden_lstm *= 2
		
		d_attn = kwargs.get('d_attn', d_hidden_lstm)
		
		# self.attention = nn.MultiheadAttention(embed_dim=d_hidden_lstm, num_heads=num_heads, dropout=dropout,
		#                                        kdim=d_attn, vdim=d_attn)
		attention_raw = kwargs.get('attention_raw', False)
		self.attention = Attention(embed_dim=d_hidden_lstm, num_heads=num_heads, dropout=dropout, kdim=d_attn, vdim=d_attn,
		                                 attention_raw=attention_raw)
		d_context = d_hidden_lstm
		d_concat = 2 * d_context
		
		d_fc_out = kwargs.get('d_fc_out', d_context)
		self.fc_squeeze = FullyConnected(d_concat, d_fc_out, activation=activation, dropout=dropout)
		
		n_fc_out = kwargs.get('n_fc_out', 0)
		self.fc_out = nn.ModuleList([
			FullyConnected(d_fc_out, d_fc_out, activation=activation, dropout=dropout) for _ in range(n_fc_out)
		])
		
		self.classifier = nn.Sequential(
			nn.Linear(d_fc_out, self.n_classes),
			nn.Dropout(p=dropout)
		)
		
		self.softmax = nn.Softmax(dim=1)
	
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
		self.lstm.flatten_parameters() # flatten parameters for data parallel
		hseq, (hidden, _) = self.lstm(x)
		
		hidden = hidden[-n_direction:].permute(1, 0, 2)  # size() == (N, n_direction, d_hidden_lstm)
		hidden = hidden.reshape(hidden.size(0), 1, -1)  # size() == (N, 1, n_direction * d_hidden_lstm)
		
		return hidden, hseq
	
	def forward(self, **input_):
		"""

		Args:
			inputs: ( input1 (N, L, h) , input2(N, L, h), optional: mask1, optional: mask2)

		Returns:

		"""
		# N = batch_size
		# L = sequence_length
		# h = hidden_dim = embedding_size
		# C = n_class
		x = input_['x']
		mask = input_['mask']
		
		# Reproduce hidden representation from LSTM
		h_last, h_seq = [torch.empty(0)] * 2, [torch.empty(0)] * 2
		for i in range(2):
			# h_last.size() == (N, 1, d_out_lstm)
			# h_seq.size() == (N, L, d_out_lstm)
			h_last[i], h_seq[i] = self.forward_lstm(x[i])
			
			# Reswapping dimension for multihead attention
			h_last[i] = h_last[i].permute(1, 0, 2)  # (1, N, d_out_lstm)
			h_seq[i] = h_seq[i].permute(1, 0, 2)  # (L, N, d_hidden_lstm)
		
		# Compute cross attention
		context, attn_weights = [None] * 2, [None] * 2
		for i in range(2):
			
			# context_1.size() == (N, 1, d_attention)
			# attn_weight_1.size() == (N, 1, L)
			# context[i], attn_weights[i] = self.attention(query=h_last[1-i], key=h_seq[i], value=h_seq[i], key_padding_mask=padding_mask)
			
			context[1 - i], attn_weights[i] = self.attention(h_last[1 - i], h_seq[i], h_seq[i],key_padding_mask=mask[i])
			context[1-i] = context[1-i].squeeze(dim=0)

		# concat the 2 context vector to make prediction
		x = torch.cat(context, dim=1)  # (N, d_concat)
		x = self.fc_squeeze(x)  # (N, d_fc_out)
		
		for fc in self.fc_out:
			x = fc(x)  # (N, d_fc_out) unchanged
		out = self.classifier(x)  # (N, n_class)

		attn_weights = [ attn_weights[i].squeeze(1) for i in range(2)]
		
		return out, attn_weights


class SigmoidPairLstmAttention(PairLstmAttention):
	
	def __init__(self, d_embedding: int, pretrained_embedding=None,  **kwargs):
		"""
		Delta model has a customized attention layers
		Args:
			vocab_size:
			d_hidden:
			embedding_weight: tensor
				if using space:
				```
				nlp = spacy.load('en_vectors_web_lg')
				embed_weights = torch.FloatTensor(nlp.vocab.vectors.data)
				```
		"""
		
		super(SigmoidPairLstmAttention, self).__init__()
		
		# Get model parameters
		
		# embedding layers
		freeze = kwargs.get('freeze', False)
		
		self.n_classes = kwargs.get('n_class', 3)
		dropout = kwargs.get('dropout', 0.)
		self.bidirectional = True  # force to default
		n_lstm = kwargs.get('n_lstm', 1)
		d_hidden_lstm = kwargs.get('d_hidden_lstm', -1)
		num_heads = kwargs.get('num_heads', 1)
		activation = kwargs.get('activation', 'relu')
		n_fc_out = kwargs.get('n_fc_out', 0)
		d_fc_out = kwargs.get('d_fc_out', -1)
		self.vocab = vocab
		
		softmax = kwargs.get('softmax', 'standard')
		t = kwargs.get('t', 1.)
		
		if pretrained_embedding is None:
			log.debug(f'Construct embedding from zero')
			self.embedding = nn.Embedding(len(vocab), d_embedding, padding_idx=vocab['<pad>'])
		else:
			log.debug(f'Load vector from pretraining')
			self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze, padding_idx=vocab['<pad>'])
		
		# LSTM block
		d_in_lstm = d_embedding
		d_hidden_lstm = d_hidden_lstm if d_hidden_lstm > 0 else d_in_lstm
		self.lstm = nn.LSTM(input_size=d_in_lstm, hidden_size=d_hidden_lstm, num_layers=n_lstm, batch_first=True,
		                    bidirectional=self.bidirectional, dropout=(n_lstm > 1) * dropout)
		
		if self.bidirectional: d_hidden_lstm *= 2
		
		# d_attn = d_attn if d_attn > 0 else d_hidden_lstm
		
		# self.attention = nn.MultiheadAttention(embed_dim=d_hidden_lstm, num_heads=num_heads, dropout=dropout,
		#                                        kdim=d_attn, vdim=d_attn)
		
		self.attention = Attention(embed_dim=d_hidden_lstm, num_heads=num_heads, dropout=dropout,
		                                 kdim=d_hidden_lstm, vdim=d_hidden_lstm,
		                                 softmax=softmax, t=t, attention_raw=True)
		d_context = d_hidden_lstm
		d_concat = 2 * d_context
		
		d_fc_out = d_embedding if d_fc_out < 0 else d_fc_out
		self.fc_squeeze = FullyConnected(d_concat, d_fc_out, activation=activation, dropout=dropout)
		
		self.fc_out = nn.ModuleList([
			FullyConnected(d_fc_out, d_fc_out, activation=activation, dropout=dropout) for _ in range(n_fc_out)
		])
		
		self.classifier = nn.Sequential(
			nn.Linear(d_fc_out, self.n_classes),
			nn.Dropout(p=dropout)
		)
		
		self.softmax = nn.Softmax(dim=1)

if __name__ == '__main__':
	
	from torch.nn.utils.rnn import pad_sequence
	from data.tokenizer import Tokenizer
	import spacy
	
	# === Params ===
	spacy_model = spacy.load('fr_core_news_md')
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
	tokenizer = Tokenizer(spacy_model=spacy_model, mode=2)
	
	counter = tokenizer.count_tokens(doc1 + doc2)
	
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
	
	model = DeltaModel(vocab=vocab,
	                     d_in=300, dropout=0, d_fc_lstm=-1, d_fc_attentiion=-1, d_context=-1, n_class=3, n_fc_out=0)
	
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
