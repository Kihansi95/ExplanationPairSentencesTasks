import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
from modules.logger import log


class IoU(_Loss):
	"""
	Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
	
	def __init__(self, eps:float=1e-7, normalize=None, size_average=None, reduce=None, reduction: str = 'mean',):
		super(IoU, self).__init__(size_average, reduce, reduction)
		self.eps = eps
		self.normalize = normalize
		
	def forward(self, input: Tensor, target: Tensor) -> Tensor:
		"""
		Args:
			input ():  a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
			target (): a tensor of shape [B, H, W] or [B, 1, H, W].
		"""
		if self.normalize is None:
			self.normalize = (input.abs() > 1.).any()
			if self.normalize:
				log.warn(f'Parameter $normalize$, initially `None`, is now set to `True`')
		
		if self.normalize:
			input = torch.sigmoid(input)
		
		#input = (input >= self.threshold).type(int)
		intersection = input.dot(target.float())
		union = torch.sum(input) + torch.sum(target) - intersection
		jaccard_index = ((intersection + self.eps) / (union + self.eps)).mean()
		return 1 - jaccard_index
	
if __name__ == '__main__':
	from torchmetrics import JaccardIndex
	
	custom_iou = IoU(normalize=True)
	m_iou = JaccardIndex(2)
	x = torch.tensor([0.2, 0.3, 1., 0.98, 0.5])
	y = torch.tensor([0, 0, 1, 1, 1])
	print(custom_iou(x, y))
	print(1 - m_iou(x, y))