import copy
import os
from copy import deepcopy

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm

label_id2label_str = {
    1: "matched",
    0: "mismatched"
}


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 2:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)

        return train_dataset, dev_dataset


"""
    Raw data are formatted as:
    {
        "id": datasets.Value("string"),
        "statement": datasets.Value("string"),
        "table": datasets.Value("string"),
        "label": datasets.Value("int32"),
    }
"""


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'totto_context_aligning_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    statement = extend_data["statement"].lower()
                    label_str = label_id2label_str[extend_data["label"]]
                    linear_table = extend_data["table"]

                    extend_data.update({"struct_in": linear_table.lower(),
                                        "text_in": statement.lower(),
                                        "seq_out": label_str.lower()})
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'totto_context_aligning_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    statement = extend_data["statement"].lower()
                    label_str = label_id2label_str[extend_data["label"]]
                    linear_table = extend_data["table"]

                    extend_data.update({"struct_in": linear_table.lower(),
                                        "text_in": statement.lower(),
                                        "seq_out": label_str.lower()})
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
