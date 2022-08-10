from torch import tensor
from torch.nn import Module
from transformers import BertTokenizer


class AddSepTransform(Module):
    """
    Add [SEP] between premise and hypothesis
    """

    def __init__(self):
        super(AddSepTransform, self).__init__()

    def forward(self, premise, hypothesis):
        if isinstance(premise, str):
            # batch of size 1 --> for the test part.
            return premise + ' [SEP] ' + hypothesis
        return [p + ' [SEP] ' + h for p, h in zip(premise, hypothesis)]


class BertTokenizeTransform(Module):

    def __init__(self, max_pad=150):
        super(BertTokenizeTransform, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_pad = max_pad

    def forward(self, text):
        tokens = self.tokenizer(text, padding="max_length", max_length=self.max_pad, truncation=True)
        return tokens.input_ids, tokens.attention_mask


class CustomToTensor(Module):
    def __init__(self):
        super(CustomToTensor, self).__init__()
        self.tensor = tensor

    def forward(self, input):
        """
        :param input: can be the input ids or the attention_mask
        """
        res = self.tensor(input)
        return res


class HateXPlainTokenization(Module):
    def __init__(self, text, max_pad=150):
        super(HateXPlainTokenization, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tensor = tensor
        self.tokenizer.con

    def forward(self, text, annotation):
        tokens = self.tokenizer(text, padding="max_length", max_length=self.max_pad, truncation=True)
        return {
            "input_ids": self.tensor(tokens.input_ids),
            "attention_masks": self.tensor(tokens.attention_mask)
        }
