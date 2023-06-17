from model import Net

import torch
from torch import nn

from model.layers.attention import Attention
from model.layers.fully_connected import FullyConnected
from modules.const import InputType
from modules.logger import log


class LstmLanguageModeling(Net):
	
	def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_dim, vocab_size)
	
	def forward(self, inputs, hidden=None):
		embedded = self.embedding(inputs)
		outputs, hidden = self.lstm(embedded, hidden)
		logits = self.fc(outputs)
		return logits, hidden
	


