import torch
from torch import Tensor
from torchmetrics import Metric

from modules.metrics.utils import batch_dot


def filter_scale_attention(preds: Tensor, target: Tensor, mask: Tensor, labels: Tensor):
	"""
	Filter attention by labels. Return the flatten vector and mask out the padding tokens
	Args:
		preds (Tensor): attention
		target (Tensor): annotations
		mask (Tensor): padding mask
		labels (Tensor): true label to filter

	Returns:
		flat_attention, flat_annotation
	"""
	# Only get label that is not neutral
	condition = labels > 0
	preds = preds.detach()
	mask, preds, target = mask[condition], preds[condition], target[condition]
	
	# rescale attentions
	value_max = torch.max(preds * ~mask, dim=1, keepdim=True).values
	value_min = torch.min(preds + 1e15 * mask, dim=1, keepdim=True).values
	preds = (preds - value_min) / (value_max - value_min + (value_max == value_min) * value_max)
	preds[preds < 0.] = 0.
	
	return preds[~mask], target[~mask]


class Entropy(Metric):
	
	full_state_update = False
	is_differentiable = True
	higher_is_better = False
	
	def __init__(self, normalize:bool=None, **kwargs):
		"""
		
		Args:
			normalize (bool): if True will normalize data
		"""
		super(Entropy, self).__init__(**kwargs)
		self.normalize = normalize
		self.add_state("cumulate_entropy", default=torch.tensor(0.), dist_reduce_fx="sum")
		self.add_state("n_sample", default=torch.tensor(0), dist_reduce_fx="sum")
		
	def update(self, preds: Tensor, mask: Tensor = None):
		if self.normalize is None:
			self.normalize = torch.any(torch.sum(preds, axis=1) != 1)
			
		batch_entropy = entropy(preds, mask, self.normalize)
		self.cumulate_entropy += batch_entropy.sum()
		self.n_sample += preds.size(0)
	
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
		mask = torch.ones(preds.shape, dtype=torch.float)
		mask = mask.type_as(preds)
	else:
		mask = 1 - mask.float()
	
	if normalize:
		preds = torch.softmax(preds - INF*(1. - mask), axis=1)
	
	log_preds = - torch.log((preds == 0) * EPS + preds)
	entropies = batch_dot(preds, log_preds)
	
	if average == 'micro':
		entropies = entropies.mean()
	elif average == 'sum':
		entropies = entropies.sum()
	return entropies


