# -*- encoding: utf-8 -*-
"""
@File    :   run_infer.py
@Time    :   2022/01/05 16:11:00
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :
"""

import argparse
import json
import logging

import torch
import pytorch_lightning as pl

from data import Classification_Dataset
from model_wrapper import Wrapper_Model

import sys

sys.path.append('../')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def gpus_parser(gpus):
    accelerator = None
    if gpus == "-1":
        # no cpu
        gpus = 0
    else:
        # single gpu
        gpus = [int(gpus), ]
    return gpus, accelerator


def main(args):
    gpu, _ = gpus_parser(args.device)
    ckpt_path = args.model_path

    logging.info(f'load model from <{ckpt_path}>')
    # load model
    model = Wrapper_Model.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        PTM_name_or_path=args.PTM_name_or_path,
        transfer_learning=None,
    )

    tokenizer = model.get_tokenizer()

    # load dataset
    dataset = Classification_Dataset(
        dataset_path=args.target_path,
        max_seq_len=args.max_len,
        tokenizer=tokenizer,
        dataset_type="predict",
    )
    dataloader = dataset.make_dataloader(2)
    # model wrapper convert binary classication and regression output to a ranking score
    trainer = pl.Trainer(gpus=gpu, logger=False)
    result = trainer.predict(model, dataloader)
    score = result[0]
    for x in result[1:]:
        score = torch.cat((score, x))
    index_list = torch.argmax(score, dim=1).tolist()
    score_list = score.tolist()

    # match score and ans
    info_dict = {}
    for example_id, example, score, index in zip(dataset.example_ids, dataset.raw_examples, score_list, index_list):
        info_dict[example_id] = {
            'context': example,
            'score': score,
            'index': index
        }

    f = open(args.output_path, 'w')
    json.dump(info_dict, f, ensure_ascii=False, indent=2)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=None)

    parser.add_argument("--model_path", type=str, required=True, help="trained ranking model ckpt")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--target_path", type=str, required=True, help='generated answer.jsonl')
    parser.add_argument("--PTM_name_or_path", type=str, required=True)

    args_str = """
    --device 0
    --max_len 64
    --model_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/Classification_Model/post_classification/data=Data_bz=2x1_ep=5_lr=5e-05_ae=1e-06_seed=42/checkpoints/epoch=04-step=11-val_loss=0.7118-acc=1.0000.ckpt
    --target_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/demo/Data/test.tsf
    --output_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/demo/Data/output/result.json
    --PTM_name_or_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/bert-base-chinese
    """

    args = parser.parse_args(args_str.split())
    logging.info(args)
    main(args)
