from functools import partial

import torch
from torchtext.vocab.vectors import pretrained_aliases as pretrained, FastText

pretrained.update({
	"fasttext.fr.300d": partial(FastText, language="fr"),
})

def init_uniform(vector: torch.Tensor):
	torch.rand(vector.size(), device=vector.device, out=vector)
	return vector
