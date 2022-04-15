# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Time    :   2022/01/06 12:31:47
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :
"""

import os
import re
import time
import json
import logging
from typing import Optional
from multiprocessing import Pool, cpu_count  # https://docs.python.org/3/library/multiprocessing.html
from collections import OrderedDict
from itertools import chain
import pickle
# from torch._C import dtype, int32

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader, sampler
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import pytorch_lightning as pl

import sys

sys.path.append('../')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class Classification_Dataset(Dataset):

    def __init__(self,
                 dataset_path, max_seq_len, tokenizer, dataset_type,
                 target=None,
                 ):

        super().__init__()
        self.example_ids = []
        self.raw_examples = []
        self.labels = []
        self.examples = []

        self.predict = True if dataset_type == 'predict' else False
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.max_seq_len = max_seq_len

        self.load_and_format_data()
        self.convert_tokens_to_ids()

    def load_and_format_data(self):
        if not os.path.isfile(self.dataset_path):
            # train 和 dev 给一个文件夹即可
            target = os.path.join(self.dataset_path, f"{self.dataset_type}.tsf")
        else:
            # test 的时候给文件
            target = self.dataset_path
        assert os.path.isfile(target)

        with open(target, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f.readlines()):
                line = line.strip().split('\t')
                example_id, context = line[:2]

                if not self.predict:
                    label = line[3]
                    self.labels.append(int(label))

                self.raw_examples.append(context)
                self.example_ids.append(example_id)
        logging.info(f"{self.dataset_type} dataset loaded")

    def convert_tokens_to_ids(self):
        logging.info(f"tokenizing {self.dataset_type} examples")
        # single thread processing
        for example in tqdm(self.raw_examples):
            feature_dict = self.tokenizer.encode_plus(example, max_length=self.max_seq_len, truncation=True,
                                                      padding='max_length')
            self.examples.append(feature_dict)
        # Multithread processing
        #
        logging.info(f"example: {self.raw_examples[0]}")

    def make_dataloader(self, batch_size):
        if self.dataset_type == "train":
            data_sampler = RandomSampler(self)
        else:
            data_sampler = SequentialSampler(self)

        dataloader = DataLoader(
            self, sampler=data_sampler,
            batch_size=batch_size, num_workers=4,
        )

        return dataloader

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feature_dict = self.examples[idx]
        for key, value in feature_dict.items():
            feature_dict[key] = torch.tensor(value)

        if not self.predict:
            feature_dict['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return feature_dict
