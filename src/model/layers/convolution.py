from typing import Union, Tuple

from torch import nn

from model.layers.activation import activate_map


class Conv(nn.Module):
	
	def __init__(self,
	    in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[str, Union[int, Tuple[int, int]]] = 0,
        activation: str or callable = 'relu',
	    **kwargs):
		
		super().__init__()
		
		if isinstance(activation, str):
			activation = activate_map[activation.lower()]
		
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=kernel_size,
				padding=padding,
				**kwargs
			),
			activation
		)
	
	def forward(self, x):
		return self.conv(x)