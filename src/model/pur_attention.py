from collections import OrderedDict

import torch
from torch import nn

from src.model.layers.attention import Attention
from src.model.layers.fully_connected import FullyConnected
from src.modules.logger import log


class PureAttention(nn.Module):

    def __init__(self, d_embedding: int,
                 padding_idx: int,
                 vocab_size: int = None,
                 pretrained_embedding=None,
                 n_class=3,
                 **kwargs):
        """
        Delta model has a customized attention layers
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

        # the attention block for multiple layers
        attention_raw = kwargs.get('attention_raw', False)
        self.attention_layers = nn.ModuleDict({
            f"att_layer_{i + 1}": Attention(embed_dim=d_embedding,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            kdim=d_embedding,
                                            vdim=d_embedding,
                                            batch_first=True,
                                            attention_raw=attention_raw)
            for i in range(num_layers)
        })

        self.classifier = nn.Sequential(
            nn.Linear(d_embedding, self.n_classes),
            nn.Dropout(p=dropout)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, **input_):
        """

        Args:
            inputs: (input (N, L, h), optional: mask1)

        Returns:

        """
        # N = batch_size
        # L = sequence_length
        # h = hidden_dim = embedding_size
        # C = n_class
        # note : always put the batch first.
        x = input_['ids']  # of shape (N, L)
        mask = input_.get('mask', torch.zeros_like(x))

        x = self.embedding(x)  # shape of (N, L, h)

        attention_weights = []
        hidden_states = [x]  # first we put the non-contextualized embeddings

        for k in self.attention_layers:
            # Compute attention
            # compute the attention on the embeddings
            # TODO : ask duc-hau for this part
            context, attn_weights = self.attention_layers[k](query=x,
                                                             key=x,
                                                             value=x,
                                                             key_padding_mask=mask)
            hidden_states.append(context)
            attention_weights.append(attn_weights)
            x = context  # update the different embeddings

        cls_tokens = x[:, 0, :]

        logits = self.classifier(cls_tokens)

        """x = torch.cat([context, h_last], dim=1)
        x = self.fc_squeeze(x)  # (N, d_fc_out)

        for fc in self.fc_out:
            x = fc(x)  # (N, d_fc_out) unchanged
        out = self.classifier(x)  # (N, n_class)

        attn_weights = attn_weights.squeeze(1)"""

        return {"last_hidden_states": x,
                "hidden_states": hidden_states,
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
