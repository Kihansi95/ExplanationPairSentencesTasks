import DataModules.transforms as t
import pytorch_lightning as pl
import torch
import os
import pandas as pd
import zipfile
from zipfile import ZipFile
import wget
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

tk = BertTokenizer.from_pretrained('bert-base-uncased')

DIR = os.path.join(".cache", "raw_data", "e_snli", "cleaned_data")
TRAIN = ["1.csv", "2.csv"]
DEV = ["dev.csv"]
TEST = ["test.csv"]

KEEP_COLS = ["premise", "hypothesis", "label", "hg_goal"]
MAX_PAD = 150

LAB_ENC = {"entailment": 0,
           "neutral": 1,
           "contradiction": 2}


def download_e_snli_raw(cache_path):
    """
    download the data into the cache_path folder
    """
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    urls = [r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_1.csv",
            r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_2.csv",
            r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv",
            r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_dev.csv"]

    for url in urls:
        nm = url.split("/")[-1]
        if os.path.exists(os.path.join(cache_path, nm)): continue
        df = pd.read_csv(url)
        df.to_csv(os.path.join(cache_path, nm))

def _sent_tokenize(sent: list):
    """
    tokenize a sentence and get it's score
    """
    buff = sent.copy()
    tok_res = []
    hg_res = []
    for w in buff:
        # if the word is higlighted
        t = None
        if "*" in w:
            # the word is higlighted
            t = tk(w.replace("*", "")).input_ids
            # remove the special tokens
            _ = t.pop(0)
            _ = t.pop(-1)

            hg_res += [1] * len(t)
        else:
            t = tk(w).input_ids
            _ = t.pop(0)
            _ = t.pop(-1)

            hg_res += [0] * len(t)

        tok_res += t

    return list(tk.convert_ids_to_tokens(torch.tensor(tok_res).detach().numpy())), \
           hg_res


def _reformat_csv(data: pd.DataFrame):
    """
    Remove unecessary columns, rename columns for better understanding. Notice that we also remove extra explanation
    columns.
    Args: data (pandas.DataFrame): Original data given by eSNLI dataset

    Returns:
        (pandas.DataFrame) clean data
    """

    rename_cols = {
        'Sentence1': 'premise',
        'Sentence2': 'hypothesis',
        'gold_label': 'label',
        'Explanation_1': 'explanation',
        'Sentence1_marked_1': 'highlight_premise',
        'Sentence2_marked_1': 'highlight_hypothesis'
    }

    drop_cols = ['pairID', 'WorkerId'
                           'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1',
                 'Explanation_2', 'Sentence1_marked_2', 'Sentence2_marked_2',
                 'Sentence1_Highlighted_2', 'Sentence2_Highlighted_2',
                 'Explanation_3', 'Sentence1_marked_3', 'Sentence2_marked_3',
                 'Sentence1_Highlighted_3', 'Sentence2_Highlighted_3']

    if data.isnull().values.any():
        data = data.dropna().reset_index()

    # rename column
    data = data.rename(
        columns=rename_cols
        # drop unneeded
    ).drop(
        columns=drop_cols, errors='ignore'
    )[['premise', 'hypothesis', 'label', 'explanation', 'highlight_premise', 'highlight_hypothesis']]

    def correct_quote(txt, hl):
        """
        Find the incoherent part in text and replace the corresponding in highlight part
        """

        # find different position between the 2
        diff = [i for i, (l, r) in enumerate(zip(txt, hl.replace('*', ''))) if l != r]
        # convert into list to be able to modify character
        txt, hl = list(txt), list(hl)
        idx = 0
        for pos_c, c in enumerate(hl):
            if c == '*': continue
            if idx in diff: hl[pos_c] = txt[idx]
            idx += 1

        hl = ''.join(hl)
        return hl

    # correct some error
    for side in ['premise', 'hypothesis']:
        data[side] = data[side].str.strip() \
            .str.replace('\\', '', regex=False) \
            .str.replace('*', '', regex=False)
        data[f'highlight_{side}'] = data[f'highlight_{side}'] \
            .str.strip() \
            .str.replace('\\', '', regex=False) \
            .str.replace('**', '*', regex=False)  # only one highlight

        # replace all the simple quote (') by double quote (") as orignal phrases
        idx_incoherent = data[side] != data[f'highlight_{side}'].str.replace('*', '', regex=False)
        sub_data = data[idx_incoherent]
        replacement_hl = [correct_quote(txt, hl) for txt, hl in
                          zip(sub_data[side].tolist(), sub_data[f'highlight_{side}'].tolist())]
        data.loc[idx_incoherent, f'highlight_{side}'] = replacement_hl

    # add some tokenization to the dataset
    for side in ['premise', 'hypothesis']:
        new_col_names = [f"tok_{side}", f"goal_{side}"]
        buff_1 = []
        buff_2 = []

        for i in range(data.shape[0]):
            sent = data[f"highlight_{side}"].values[i].split(" ")
            tok_res, hg_res = _sent_tokenize(sent)
            buff_1.append(tok_res)
            buff_2.append(hg_res)

        # add the columns to the dataframe
        data[new_col_names[0]] = buff_1.copy()
        data[new_col_names[1]] = buff_2.copy()

    return data


def _combine_sentences(data: pd.DataFrame):
    data['CLS'] = [["[CLS]"]] * data.shape[0]
    data['SEP'] = [["[SEP]"]] * data.shape[0]
    data['BUFF'] = [[0]] * data.shape[0]

    data['tok_sent'] = data['CLS'] + data['tok_premise'] + data['SEP'] + data['tok_hypothesis'] + data['SEP']
    data['hg_goal'] = data['BUFF'] + data['goal_premise'] + data['BUFF'] + data['goal_hypothesis'] + data['BUFF']

    drop_columns = ['tok_premise', 'goal_premise',
                    'tok_hypothesis', 'goal_hypothesis',
                    'CLS', 'BUFF', 'SEP']

    return data


# download the data for the training
def process_e_snli_data(cache_path):
    """
    process the e_snli_data present in the cache_path folder
    +
    download them to the cleaned data folder
    """
    files = ["esnli_" + u for u in ["dev.csv", "test.csv", "train_1.csv", "train_2.csv"]]
    dirs = [os.path.join(cache_path, f) for f in files]
    save_dir = os.path.join(cache_path, "cleaned_data")

    if not (os.path.exists(save_dir)):
        # creation of the save dir
        os.mkdir(save_dir)

    for d in dirs:
        sv_dir = os.path.join(save_dir, d.split("_")[-1])
        if os.path.exists(sv_dir): continue
        df = pd.read_csv(d, sep=",")
        df = _reformat_csv(df)
        df = _combine_sentences(df)
        df.to_csv(sv_dir)

###################
### The DataSet ###
###################

class EsnliDataSet(Dataset):
    def __init__(self, split="TRAIN", nb_data=-1, cache_path=DIR):
        self.dirs = [os.path.join(cache_path, f) for f in eval(split)]  # where the datas are
        self.data = None
        if split == "TRAIN":
            # in the function don't forget to split for the training and validation part.
            df1 = pd.read_csv(self.dirs[0], usecols=KEEP_COLS)
            df2 = pd.read_csv(self.dirs[1], usecols=KEEP_COLS)
            self.data = pd.concat([df1, df2])
        elif split in ["TEST", "DEV"]:
            self.data = pd.read_csv(os.path.join(self.dirs[0]), usecols=KEEP_COLS)

        if nb_data > 0:
            # fraction of data to keep
            frac = nb_data / self.data.shape[0]
            self.data = self.data.sample(frac=frac).reset_index()

    def __getitem__(self, item):
        premise = self.data.premise.values[item]
        hypothesis = self.data.hypothesis.values[item]
        buff = eval(self.data.hg_goal.values[item])
        annotation = buff + [0] * (MAX_PAD - len(buff))
        label = self.data.label.values[item]
        return {"premise": premise, "hypothesis": hypothesis,
                "annotation": torch.tensor(annotation, requires_grad=False),
                "label": torch.tensor(LAB_ENC[label])}

    def __len__(self):
        return self.data.shape[0]

class ESNLIDataModule(pl.LightningDataModule):
    """
    Data module (pytorch lightning) for the SNLI dataset

    Attributes :
    ------------
        cache : the location of the data on the disk
        batch_size : batch size for the training
        num_workers : number of cpu heart to use for using the data
        nb_data : number of data for the training
        t_add_sep : modules class to add the [SEP] token
        t_tokenize : modules class to return the attention_masks and input ids
        t_tensor : modules class to transform the input_ids and att_mask into tensors
    """

    def __init__(self, cache: str, batch_size=8, num_workers=0, nb_data=-1):
        super().__init__()
        self.cache = cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nb_data = nb_data

        self.t_add_sep = t.AddSepTransform()
        self.t_tokenize = t.BertTokenizeTransform(max_pad=150)
        self.t_tensor = t.CustomToTensor()

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        download_e_snli_raw(self.cache)
        process_e_snli_data(self.cache)

    def setup(self, stage: str = None):
        """
        set_up function :

            - this function will prepare the data by setting the differents attributes self.train_set ... etc

        :param stage : are we preparing the data for the training part or the test part.
        """

        # called on every GPU
        # load dataset from cache in each instance of GPU
        if stage == 'fit' or stage is None:
            buff = None
            if self.nb_data > 0:
                buff = EsnliDataSet(split="TRAIN", nb_data=self.nb_data,
                                    cache_path=os.path.join(self.cache, "cleaned_data"))
            else:
                buff = EsnliDataSet(split="TRAIN", nb_data=-1, cache_path=os.path.join(self.cache, "cleaned_data"))
            # 80% train 20% validation
            train_size = int(0.8 * len(buff))
            val_size = len(buff) - train_size
            self.train_set, self.val_set = torch.utils.data.random_split(buff, [train_size, val_size])

        if stage == 'test' or stage is None:
            buff = None
            if self.nb_data > 0:
                buff = EsnliDataSet(split="TEST", nb_data=self.nb_data,
                                    cache_path=os.path.join(self.cache, "cleaned_data"))
            else:
                buff = EsnliDataSet(split="TEST", nb_data=-1, cache_path=os.path.join(self.cache, "cleaned_data"))
            self.test_set = buff

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    ## ======= PRIVATE SECTIONS ======= ##
    def collate(self, batch):
        batch = self.list2dict(batch)
        texts = self.t_add_sep(batch['premise'], batch['hypothesis'])
        input_ids, attention_mask = self.t_tokenize(texts)
        input_ids = self.t_tensor(input_ids)
        attention_mask = self.t_tensor(attention_mask)
        annotation = torch.stack(batch["annotation"], dim=0)
        labels = self.t_tensor(batch['label'])

        return {"input_ids": input_ids, "attention_masks": attention_mask, "labels": labels, "annotations": annotation}

    # maybe not usefull here
    def list2dict(self, batch):
        # convert list of dict to dict of list
        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}
        return {k: [row[k] for row in batch] for k in batch[0]}