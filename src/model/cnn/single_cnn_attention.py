import torch
from torch import nn
import torch.nn.functional as F

from model.layers.attention import Attention
from model.layers.fully_connected import FullyConnected
from modules.logger import log


class SingleCnnAttention(nn.Module):
	
	def __init__(self, d_embedding: int, padding_idx: int, vocab_size:int=None, pretrained_embedding=None, n_class=3, **kwargs):
		"""
		Delta model has a customized attention layers
		"""
		
		super(SingleCnnAttention, self).__init__()
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
		
		# contextualization block
		d_in_context = d_embedding
		c_out = 100
		kernel_size = 5
		n_kernel = kwargs.get('n_kernel', 1)
		d_out_context = c_out * n_kernel
		n_context = kwargs.get('n_context', 1) # TODO
		
		self.conv = nn.Conv2d(
			in_channels=1,
			out_channels=c_out,
			kernel_size=(kernel_size, d_in_context),
			padding=(kernel_size//2,0)
		)
		self.relu = nn.ReLU()
		
		d_attn = kwargs.get('d_attn', d_out_context)
		
		attention_raw = kwargs.get('attention_raw', False)
		self.attention = Attention(embed_dim=d_out_context, num_heads=num_heads, dropout=dropout, kdim=d_attn, vdim=d_attn, batch_first=True, attention_raw=attention_raw)
		d_context = d_out_context
		
		d_fc_out = kwargs.get('d_fc_out', d_context)
		self.fc_squeeze = FullyConnected(d_context + d_out_context, d_fc_out, activation=activation, dropout=dropout)
		
		n_fc_out = kwargs.get('n_fc_out', 0)
		self.fc_out = nn.ModuleList([
			FullyConnected(d_fc_out, d_fc_out, activation=activation, dropout=dropout) for _ in range(n_fc_out)
		])
		
		self.classifier = nn.Sequential(
			nn.Linear(d_fc_out, self.n_classes),
			nn.Dropout(p=dropout)
		)
		
		self.softmax = nn.Softmax(dim=1)
		
		
	
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
		x = input_['ids']
		mask = input_.get('mask', torch.zeros_like(x))
		
		# Reproduce hidden representation from CNN
		x = self.embedding(x)                       # (B, L, h)
		x = x.unsqueeze(1)                          # (B, C_in, L, h)
		
		h_seq = self.conv(x)                        # (B, C_out, L, 1)
		h_seq = self.relu(h_seq)
		h_seq = h_seq.squeeze(-1)                   # (B, C_out, L)
		
		# Get context vector by using max_pooling
		h_context = F.max_pool1d(h_seq, h_seq.size(-1))  # (B, C_out, 1)
		#h_context = h_context.squeeze(-1)           # (B, C_out)
		
		# Reswapping dimension for attention
		h_seq = h_seq.permute(0,2,1)                # (B, L, C_out)
		h_context = h_context.permute(0,2,1)        # (B, 1, C_out)
		
		# Compute attention
		# context.size() == (N, 1, d_attention)
		# attn_weight.size() == (N, 1, L)
		context, attn_weights = self.attention(query=h_context, key=h_seq, value=h_seq, key_padding_mask=mask)
		context = context.squeeze(dim=1)
		h_context = h_context.squeeze(dim=1)
		# x = context.squeeze(dim=1)

		x = torch.cat([context, h_context], dim=1)
		x = self.fc_squeeze(x)  # (N, d_fc_out)
		
		for fc in self.fc_out:
			x = fc(x)  # (N, d_fc_out) unchanged
		out = self.classifier(x)  # (N, n_class)
		
		attn_weights = attn_weights.squeeze(1)
		
		return out, attn_weights


if __name__ == '__main__':
	
	import spacy
	from torch import optim
	from torch.nn.utils.rnn import pad_sequence
	from torchtext.vocab import Vocab
	from torchtext.data import get_tokenizer
	
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
	# tokenizer = Tokenizer(spacy_model=spacy_model, mode=2)
	tokenizer = get_tokenizer('spacy') # spacy tokenizer, provided by torchtext
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
	
	model = SingleCnnAttention(d_in=300, dropout=0, d_fc_attentiion=-1, d_context=-1, n_class=3, n_fc_out=0)
	
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
