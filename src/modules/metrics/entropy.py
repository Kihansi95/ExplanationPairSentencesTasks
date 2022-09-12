import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.enums import AverageMethod

from modules.metrics.utils import batch_dot

_SUM = 'sum'

class Entropy(Metric):
	
	full_state_update = False
	is_differentiable = True
	higher_is_better = False
	
	def __init__(self, normalize:bool=None, average:str=None, **kwargs):
		"""
		
		Args:
			normalize (bool): if True will normalize data
		"""
		super(Entropy, self).__init__(**kwargs)
		self.normalize = normalize
		self.average = average
		self.add_state("cumulate_entropy", default=torch.tensor(0.), dist_reduce_fx="sum")
		self.add_state("n_sample", default=torch.tensor(0), dist_reduce_fx="sum")
		
	def update(self, preds: Tensor, mask: Tensor = None):
		if self.normalize is None:
			self.normalize = torch.any(torch.sum(preds, axis=1) != 1)
			
		batch_entropy = entropy(preds, mask, self.normalize, self.average)
		if self.average == None:
			self.cumulate_entropy += batch_entropy.sum()
			self.n_sample += preds.size(0)
			
		elif self.average == _SUM:
			self.cumulate_entropy += batch_entropy
			self.n_sample += preds.size(0)
			
		elif self.average == AverageMethod.MICRO:
			self.cumulate_entropy += batch_entropy
			self.n_sample += 1
	
	def compute(self):
		return self.cumulate_entropy.float() / self.n_sample
	
EPS = 1e-10
INF = 1e30
def entropy(preds: Tensor, mask: Tensor=None, normalize:bool=None, average:str=None) -> Tensor:
	"""
	
	Args:
		preds (tensor): batch of vector dim-D (BxD)
		mask (tensor): boolean or float. value (0.) where we do not want to take into entropy
		normalize (bool): If need to renormalize distribution
		average (str): If none, do not average. If average = 'micro', will take the mean over batch

	Returns:

	"""
	if mask is None:
		mask = torch.ones(preds.shape, dtype=torch.float).type_as(preds)
	else:
		mask = 1 - mask.float()
	
	if normalize:
		preds = torch.softmax(preds - INF*(1. - mask), axis=1)
	
	log_preds = - torch.log((preds == 0) * EPS + preds)
	log_length = torch.log(mask.sum(axis=1)).float()
	
	entropies = batch_dot(preds, log_preds) / log_length

	if average == AverageMethod.MICRO:
		entropies = entropies.mean()
	elif average == _SUM:
		entropies = entropies.sum()
	return entropies


