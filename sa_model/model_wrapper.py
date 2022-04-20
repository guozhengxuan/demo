"""
ref
https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/text-transformers.ipynb#scrollTo=ddfafe98
https://pytorch-lightning.readthedocs.io/en/latest/
"""
import logging

import torch
import torch.nn.functional as F
from sklearn import metrics
from pytorch_lightning import LightningModule

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    BertTokenizer, BertForSequenceClassification, BertConfig
)


class Wrapper_Model(LightningModule):
    def __init__(
        self,
        PTM_name_or_path,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_proportion: int = 0,
        weight_decay: float = 0.0,
        train_batch_size_pre_device: int = 32,
        args_str: str = '',
        problem_type = '',
        label_num = 4,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        Toknizer = BertTokenizer
        Model = BertForSequenceClassification
        Config = BertConfig

        self.config = Config.from_pretrained(PTM_name_or_path)
        self.config.problem_type = "single_label_classification"
        self.config.num_labels = self.hparams.label_num

        self.model = Model.from_pretrained(PTM_name_or_path, config=self.config)
        self.tokenizer = Toknizer.from_pretrained(PTM_name_or_path)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        predicts = torch.argmax(logits, 1)
        return {'loss': val_loss, 'predicts': predicts, 'labels': batch['labels']}

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()

        predicts = validation_step_outputs[0]['predicts']
        labels = validation_step_outputs[0]['labels']
        for x in validation_step_outputs[1:]:
            predicts = torch.cat((predicts, x['predicts']))
            labels = torch.cat((labels, x['labels']))
        acc = metrics.accuracy_score(labels.cpu(), predicts.cpu())

        self.log("val_loss", loss, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        score = F.softmax(outputs[0], dim=1)
        return score

    def set_example_num(self, example_num):
        self.example_num = example_num

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        if isinstance(self.trainer.gpus, list):
            gpus = len(self.trainer.gpus)
        else:
            gpus = self.trainer.gpus

        tb_size = self.hparams.train_batch_size_pre_device * max(1, gpus)
        ab_size = self.trainer.accumulate_grad_batches
        total_steps = int((self.example_num * int(self.trainer.max_epochs) // tb_size) // ab_size )
        warmup_steps = int(self.hparams.warmup_proportion * total_steps)
        logging.info(f'total_steps: {total_steps}; warmup_steps: {warmup_steps}')

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        if self.hparams.get('cosine_schedule', False):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
