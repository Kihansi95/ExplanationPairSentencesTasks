from model import Net

import torch
from torch import nn
import torch.nn.functional as F

from model.layers.attention import Attention
from model.layers.convolution import ConvMultiKernel
from model.layers.fully_connected import FullyConnected
from modules.logger import log


class DualCnnAttention(Net):
	
	def __init__(self, d_embedding: int, padding_idx: int, vocab_size:int=None, pretrained_embedding=None, n_class=3, **kwargs):
		"""
		Delta model has a customized attention layers
		"""
		
		super(DualCnnAttention, self).__init__()
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
			self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=padding_idx)
		else:
			log.debug(f'Load vector from pretraining')
			self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=kwargs.get('freeze', False), padding_idx=padding_idx)
		
		d_embedding = self.embedding.embedding_dim
		
		# contextualization block
		d_in_context = d_embedding
		c_out = 100
		n_kernel = kwargs.get('n_kernel', 3)
		d_out_context = c_out * n_kernel
		# Bidirectional
		
		self.conv = ConvMultiKernel(
			in_channels=1,
			out_channels=c_out,
			kernels=n_kernel,
			feature_dim=d_in_context,
			activation='relu'
		)
		
		d_attn = kwargs.get('d_attn', d_out_context)
		
		attention_raw = kwargs.get('attention_raw', False)
		self.attention = Attention(embed_dim=d_out_context, num_heads=num_heads, dropout=dropout, kdim=d_attn, vdim=d_attn, batch_first=True, attention_raw=attention_raw)
		d_context = d_out_context
		d_concat = 2 * d_context
		
		d_fc_out = kwargs.get('d_fc_out', d_context)
		self.fc_squeeze = FullyConnected(d_concat, d_fc_out, activation=activation, dropout=dropout)
		
		n_fc_out = kwargs.get('n_fc_out', 0)
		self.fc_out = nn.ModuleList([ FullyConnected(d_fc_out, d_fc_out, activation=activation, dropout=dropout) for _ in range(n_fc_out)])
		
		self.classifier = nn.Sequential(
			nn.Linear(d_fc_out, self.n_classes),
			nn.Dropout(p=dropout)
		)
		
		self.softmax = nn.Softmax(dim=1)
		
	
	def forward(self, premise_ids, hypothesis_ids, premise_padding=None, hypothesis_padding=None):
		"""

		Args:
			inputs: ( input1 (N, L, h) , input2(N, L, h), optional: mask1, optional: mask2)

		Returns:

		"""
		# N = batch_size
		# L = sequence_length
		# h = hidden_dim = embedding_size
		# C = n_class
		#ids = [premise_ids, hypothesis_ids]
		ids = {'premise': premise_ids, 'hypothesis': hypothesis_ids}
		padding_mask = {'premise': premise_padding, 'hypothesis': hypothesis_padding}
		# mask = [premise_padding, hypothesis_padding]
		
		# Reproduce hidden representation from LSTM
		h_context = dict()
		h_seq = dict()
		for s, x in ids.items():
			x = self.embedding(x)  # (B, L, h)
			x = x.unsqueeze(1)  # (B, C_in, L, h)
			
			# Contextualize
			h_seq[s] = self.conv(x)  # (B, n_kernel * C_out, L)
			
			# Get context vector by using max_pooling
			h_context[s] = F.max_pool1d(h_seq[s], h_seq[s].size(-1))
			
			# Reswapping dimension for attention
			h_seq[s] = h_seq[s].permute(0, 2, 1)  # (B, L, C_out)
			h_context[s] = h_context[s].permute(0, 2, 1)  # (B, 1, C_out)
			
		# Compute cross attention
		attn_weights = dict()
		context = dict()
		sides = list(ids.keys())
		for s, s_bar in sides, sides[::-1]:
			# context.size() == (N, 1, d_attention)
			# attn_weight.size() == (N, 1, L)
			context[s_bar], attn_weights[s] = self.attention(query=h_context[s_bar], key=h_seq[s], value=h_seq[s],key_padding_mask=padding_mask[s])
			context[s_bar] = context[s_bar].squeeze(dim=1)  # context.size() == (N, d_attention)
			attn_weights[s] = attn_weights[s].squeeze(1)    # attn_weights.size() == (N, L)
		
		# concat the 2 context vector to make prediction
		x = torch.cat(list(context.values()), dim=1)  # (N, d_concat)
		x = self.fc_squeeze(x)  # (N, d_fc_out)
		
		for fc in self.fc_out:
			x = fc(x)  # (N, d_fc_out) unchanged
		out = self.classifier(x)  # (N, n_class)
		
		return out, attn_weights


if __name__ == '__main__':
	
	from torch.nn.utils.rnn import pad_sequence
	from torchtext.vocab import build_vocab_from_iterator
	from torch import nn, optim
	import spacy
	
	# === Params ===
	spacy_model = spacy.load('en_core_web_sm')
	
	# === Examples ===
	doc1 = [
		'A man inspects the uniform of a figure in some East Asian country.',
		'An older and younger man smiling.',
		'A black race car starts up in front of a crowd of people.'
	]
	doc2 = [
		'The man is sleeping',
		'Two men are smiling and laughing at the cats playing on the floor.',
		'A man is driving down a lonely road.'
	]
	y = [0, 1, 2]
	
	# Tokenize
	# ==============
	# build vocab
	tokens_1 = [[tk.lemma_.lower() for tk in doc] for doc in spacy_model.pipe(doc1)]
	tokens_2 = [[tk.lemma_.lower() for tk in doc] for doc in spacy_model.pipe(doc2)]
	
	vocab = build_vocab_from_iterator(tokens_1 + tokens_2, specials=['<unk>', '<pad>', '<msk>'])

	# === Test ===
	
	# tokenize
	x1 = [vocab(tks) for tks in tokens_1]
	x2 = [vocab(tks) for tks in tokens_2]
	
	# convert to tensor
	x1 = [torch.tensor(x, dtype=torch.long) for x in x1]
	x2 = [torch.tensor(x, dtype=torch.long) for x in x2]
	
	x1 = pad_sequence(x1, batch_first=True)
	x2 = pad_sequence(x2, batch_first=True)
	
	y = torch.tensor(y, dtype=torch.long)
	padding_idx = vocab['<pad>']
	model = DualCnnAttention(d_embedding=300,
	                         padding_idx=padding_idx,
	                         vocab_size=len(vocab),
	                         dropout=0,
	                         d_fc_lstm=-1,
	                         d_fc_attentiion=-1,
	                         d_context=-1,
	                         n_class=3,
	                         n_fc_out=0)
	
	model.train()
	
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	
	for epoch in range(1):
		# reset optimizer
		optimizer.zero_grad()
		
		preds, _ = model(premise_ids=x1, hypothesis_ids=x2, premise_padding=x1==padding_idx, hypothesis_padding=x2==padding_idx)
		loss = loss_fn(preds, y)
		
		loss.backward()
		optimizer.step()
		
		running_loss = loss.item()
		print("[{:0>3d}] loss: {:.3f}".format(epoch + 1, running_loss))
	
	model.eval()
	predict, _ = model(premise_ids=x1, hypothesis_ids=x2, premise_padding=x1==padding_idx, hypothesis_padding=x2==padding_idx)
	predict = predict.detach()
	print('Prediction:')
	print(predict)
