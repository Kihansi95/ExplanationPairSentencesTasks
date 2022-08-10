import torch

INF = 1e30 # Infinity

def rescale(attention: torch.Tensor, mask: torch.Tensor):
	v_max = torch.max(attention + mask.float() * -INF, dim=1, keepdim=True).values
	v_min = torch.min(attention + mask.float() * INF, dim=1, keepdim=True).values
	v_min[v_min == v_max] = 0.
	rescale_attention = (attention - v_min)/(v_max - v_min)
	rescale_attention[mask] = 0.
	return rescale_attention