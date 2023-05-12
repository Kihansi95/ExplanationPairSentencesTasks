from typing import Union


def mean(x: Union[dict, list]):
	"""Average values in x array
	
	Parameters
	----------
	x : list or dictionary
	
	Returns
	-------
	mean_value : float
		If `x` is a dictionary, `mean_value` is average of all values in x. If `x` is a list, `mean_value` is the average of all element in list.
	
	Raises
	-------
	NotImplementedError
		If x is neither list nor dictionary.
	
	"""
	if isinstance(x, dict):
		return sum(x.values()) / len(x)
	if isinstance(x, list):
		return sum(x) / len(x)
	
	raise NotImplementedError()