import argparse
import shutil
import logging
import pickle

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model_wrapper import Wrapper_Model
from data import Classification_Dataset
from cfg import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('')


def gpus_parser(gpus):
    accelerator = None
    if gpus == "-1":
        # no cpu
        gpus = 0
    else:
        # single gpu
        gpus = [int(gpus), ]
    return gpus, accelerator


def get_version_name(args):
    version_name = ''
    temp = os.path.basename(args.dataset_path)
    version_name += f"data={temp}"
    version_name += f"_bz={args.train_batch_size_per_gpu}x{args.gradient_accumulation_step}"
    version_name += f"_ep={args.epoch}_lr={args.learning_rate}_ae={args.adam_epsilon}_seed={args.seed}"
    return version_name


def set_logger(args, version_name):
    tblogger = TensorBoardLogger(
        args.output_path,
        name=args.task_name,
        version=version_name
    )
    return tblogger


def main(args):
    seed_everything(args.seed)
    version = get_version_name(args)
    tblogger = set_logger(args, version)
    logger.info(str(args))

    # model & tokenizer init
    model = Wrapper_Model(
        PTM_name_or_path=args.PTM_name_or_path,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        train_batch_size_pre_device=args.train_batch_size_per_gpu,
        args_str=str(args),
        label_num=args.label_num
    )
    tokenizer = model.get_tokenizer()

    # dataset -> dataloader
    logger.info(f"load dataset from {args.dataset_path}")
    if not os.path.exists(SA_TRAIN_CACHE):
        train_dataset = Classification_Dataset(
            args.dataset_path, args.max_len, tokenizer, "train",
        )
        with open(SA_TRAIN_CACHE, 'wb') as f:
            pickle.dump(train_dataset, f)
    else:
        with open(SA_TRAIN_CACHE, 'rb') as f:
            train_dataset = pickle.load(f)

    dev_dataset = Classification_Dataset(
        args.dataset_path, args.max_len, tokenizer, "dev",
    )

    train_dataloader = train_dataset.make_dataloader(
        batch_size=args.train_batch_size_per_gpu
    )
    dev_dataloader = dev_dataset.make_dataloader(
        batch_size=args.dev_batch_size_per_gpu)
    model.set_example_num(len(train_dataset))

    # set trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='acc',
        every_n_epochs=1,
        filename="{epoch:02d}-{step}-{val_loss:.4f}-{acc:.4f}",
        save_top_k=4,
        mode="max",
        save_weights_only=True
    )

    gpus, accelerator = gpus_parser(args.gpus_id)
    trainer = Trainer(
        max_epochs=args.epoch,
        val_check_interval=0.1,
        gpus=gpus,
        accelerator=accelerator,
        deterministic=True,
        default_root_dir=args.output_path,
        logger=tblogger,
        accumulate_grad_batches=args.gradient_accumulation_step,
        callbacks=[checkpoint_callback, ],
    )

    trainer.fit(model, train_dataloader)
    trainer.fit(model, train_dataloader, dev_dataloader)

    if not os.path.exists(SA_BEST_MODEL):
        os.mkdir(SA_BEST_MODEL)
    shutil.copy(checkpoint_callback.best_model_path, SA_BEST_MODEL)

    logger.info('finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path and name
    parser.add_argument("--PTM_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)

    # dataset & model mode
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--label_num", type=int, default=0)

    # hparams & device=
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_proportion", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--train_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--gradient_accumulation_step", default=1, type=int)
    parser.add_argument("--dev_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpus_id", type=str, help='not use -1, use 0')

    arg_str = f"""
    --task_name post_classification
    --label_num 4
    --PTM_name_or_path {BERT}
    --output_path {SA_MODEL}
    --dataset_path {SA_DATASET}
    --max_len 64
    --epoch 1
    --learning_rate 5e-5
    --adam_epsilon 1e-6
    --warmup_proportion 0.01
    --train_batch_size_per_gpu 4
    --gradient_accumulation_step 2
    --dev_batch_size_per_gpu 32
    --seed 42
    --gpus_id -1
    """

    args = parser.parse_args(arg_str.split())

    main(args)
