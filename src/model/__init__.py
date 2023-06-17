from abc import ABC
from torch import nn


class Net(nn.Module, ABC):
	
	def __init__(self):
		super().__init__()
		
	def __str__(self):
		return self.__class__.__name__