import torch

INF = 1e30 # Infinity

def rescale(attention: torch.Tensor, mask: torch.Tensor):
	v_max = torch.max(attention + mask.float() * -INF, dim=1, keepdim=True).values
	v_min = torch.min(attention + mask.float() * INF, dim=1, keepdim=True).values
	v_min[v_min == v_max] = 0.
	rescale_attention = (attention - v_min)/(v_max - v_min)
	rescale_attention[mask] = 0.
	return rescale_attention

def hightlight_txt(txt, weights):
    """
    Build an HTML of text along its weights.
    Args:
        txt:
        weights:

    Returns: str
    Examples:
        ```python
        from IPython.core.display import display, HTML
        highlighted_text = hightlight_txt(lemma1[0], a1v2)
        display(HTML(highlighted_text))
        ```
    """
    max_alpha = 0.8

    highlighted_text = ''
    w_min, w_max = torch.min(weights), torch.max(weights)
    w_norm = (weights - w_min)/(w_max - w_min)

    for i in range(len(txt)):
        highlighted_text += '<span style="background-color:rgba(135,206,250,' \
                            + str(float(w_norm[i]) / max_alpha) + ');">' \
                            + txt[i] + '</span> '

    return highlighted_text
