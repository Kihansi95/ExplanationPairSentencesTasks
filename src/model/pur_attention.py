from collections import OrderedDict

import torch
from torch import nn
import math
from torch.nn import MultiheadAttention
from model.layers.attention import Attention
from model.layers.fully_connected import FullyConnected
from modules.logger import log

# huggin face class for the positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        temp = self.pe.repeat(x.size(0), 1, 1).clone().detach().to(x.device)
        x = x + temp[x.size(1), :]
        return self.dropout(x)


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
        # TODO : check where is the problem in the gradient.
        self.attention_layers = nn.ModuleList([
            Attention(embed_dim=d_embedding,
                      num_heads=num_heads,
                      dropout=dropout,
                      kdim=d_embedding,
                      vdim=d_embedding,
                      batch_first=True,
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
        x = self.embedding(x)  # shape of (N, L, h)

        # the positional encoding
        # x = self.pe(x)

        attention_weights = []  # each element of the list is of size (N, H, L, L)

        for i, l in enumerate(self.attention_layers):
            # Compute attention : contextualization of the embeddings
            # compute the attention on the embeddings
            # /!\ the attention weights are already averaged on the number of heads.
            x, attn_weights = l(query=x,
                                key=x,
                                value=x,
                                key_padding_mask=mask
                                )

            attention_weights.append(attn_weights)  # we add the different attention weights while we progress.

        # cls token of the last hidden state
        cls_tokens = x[:, 0, :]
        # log.debug(f"cls_tok : {cls_tokens}")

        logits = self.classifier(cls_tokens)

        return {
            "last_hidden_states": x,
            "attn_weights": attention_weights,
            "cls_tokens": cls_tokens,
            "logits": logits
        }


if __name__ == '__main__':

    import spacy
    from torch import optim
    from torch.nn.utils.rnn import pad_sequence
    from torchtext.vocab import Vocab
    from torchtext.data import get_tokenizer

    # === Params ===
    spacy_model = spacy.load('fr_core_news_md')
    method = 'general'
    h = spacy_model.vocab.vectors.shape[-1]

    # === Examples ===
    doc1 = [
        'Bonjour tonton',
        'Comment allez-vous?',
        'Nik a les cheveux courts.'
    ]
    doc2 = [
        'On l’utilise principalement entre copains, entre écoliers, entre jeunes…',
        'Ce repas/plat était très bon!',
        'Tina a les cheveux bouclés.'
    ]
    y = [0, 1, 2]

    # Tokenize
    # ==============
    # tokenizer = Tokenizer(spacy_model=spacy_model, mode=2)
    tokenizer = get_tokenizer('spacy', language="fr")  # spacy tokenizer, provided by torchtext
    counter = tokenizer.count_tokens(doc1 + doc2)

    vocab = Vocab(counter, specials=['<unk>', '<pad>', '<msk>'])
    tokenizer.vocab = vocab
    # === Test ===

    # tokenize
    x1 = [tokenizer.numericalize(d) for d in doc1]
    x2 = [tokenizer.numericalize(d) for d in doc2]

    # convert to tensor
    x1 = [torch.tensor(x, dtype=torch.long) for x in x1]
    x2 = [torch.tensor(x, dtype=torch.long) for x in x2]

    x1 = pad_sequence(x1, batch_first=True)
    x2 = pad_sequence(x2, batch_first=True)

    y = torch.tensor(y, dtype=torch.long)

    model = PureAttention(d_in=300, dropout=0, num_heads=1, num_layers=1)

    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        # reset optimizer
        optimizer.zero_grad()

        preds, _ = model([x1, x2])
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        print("[{:0>3d}] loss: {:.3f}".format(epoch + 1, running_loss))

    model.eval()
    predict, _ = model([x1, x2])
    predict = predict.detach()
    print('Prediction:')
    print(predict)
