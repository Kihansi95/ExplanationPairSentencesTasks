import torch
from model.attention.positional_encoding import PositionalEncoding
from torch import nn
from model.layers.attention_key import Attention
from modules.logger import log


class PureAttention(nn.Module):

    def __init__(self, d_embedding: int,
                 padding_idx: int,
                 vocab_size: int = None,
                 pretrained_embedding=None,
                 n_class=3,
                 **kwargs):
        """
        Args:
            d_embedding: dimension of the embeddings
            padding_idx: index of the padding token in the vocab
            vocab_size: length of the vocab
            pretrained_embedding: pre-trained vectors if we don't want to build them from zeros
            n_class: number of classes for the classification
            **kwargs: additional parameters
        """

        super(PureAttention, self).__init__()
        # Get model parameters
        assert not (
                vocab_size is None and pretrained_embedding is None
        ), 'Provide either vocab size or pretrained embedding'

        # embedding layers
        freeze = kwargs.get('freeze', False)

        self.n_classes = n_class
        dropout = kwargs.get('dropout', 0.)
        num_heads = kwargs.get('num_heads', 1)
        activation = kwargs.get('activation', 'relu')
        num_layers = kwargs.get('num_layers', 1)

        assert (num_heads >= 1), 'please put at least one head'
        assert (num_layers >= 1), 'please put at least one layer'

        if pretrained_embedding is None:
            log.debug(f'Construct embedding from zero')
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embedding, padding_idx=padding_idx)
        else:
            log.debug(f'Load vector from pretraining')
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze, padding_idx=padding_idx)

        # add a positional encoding layer
        self.pe = PositionalEncoding(d_model=d_embedding,
                                     dropout=0,
                                     max_len=10000)

        # attention layers store attention layers in module list : keep the gradient in the graph.
        attention_raw = kwargs.get('attention_raw', False)
        self.attention_layers = nn.ModuleList([
            Attention(embed_dim=d_embedding,
                      num_heads=num_heads,
                      dropout=dropout,
                      kdim=d_embedding,
                      vdim=d_embedding,
                      batch_first=True
                      )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_embedding, self.n_classes),
            nn.Dropout(p=dropout)
        )

    def forward(self, **input_):
        """
        Args:
            **input_:

        Returns:
            A dictionnary for the outputs.
        """
        # N = batch_size
        # L = sequence_length
        # h = hidden_dim = embedding_size
        # H = number of heads
        # C = n_class
        x = input_['ids']  # of shape (N, L)
        mask = input_.get('mask', torch.zeros_like(x))

        # non contextual embeddings
        hidden_states = []
        x = self.embedding(x)  # shape of (N, L, h)
        hidden_states.append(x)

        # the positional encoding
        x = self.pe(x)

        attention_weights = []
        key_embeddings = []
        request_embeddings = []
        value_embeddings = []

        for i, l in enumerate(self.attention_layers):
            # Compute attention : contextualization of the embeddings
            # compute the attention on the embeddings
            # /!\ the attention weights are already averaged on the number of heads.
            context, attn_weights, k, q, v = l(query=x,
                                               key=x,
                                               value=x,
                                               key_padding_mask=mask
                                               )

            # ADD
            x = context + x

            # NORM
            m = torch.mean(x, dim=-1).unsqueeze(dim=-1).repeat(1, 1, x.shape[-1])
            sd = torch.sqrt(torch.var(x, dim=-1).unsqueeze(dim=-1).repeat(1, 1, x.shape[-1]))
            x = (x - m) / sd

            # the embedding and the attention
            hidden_states.append(x)
            attention_weights.append(attn_weights)

            # update the embeddings on the different spaces
            key_embeddings.append(k)
            request_embeddings.append(q)
            value_embeddings.append(v)

        cls_tokens = x[:, 0, :]
        logits = self.classifier(cls_tokens)

        return {
            "last_hidden_states": x,
            "hidden_states": hidden_states,
            "attn_weights": attention_weights,
            "cls_tokens": cls_tokens,
            "key_embeddings": key_embeddings,
            "request_embeddings": request_embeddings,
            "value_embeddings": value_embeddings,
            "logits": logits
        }
