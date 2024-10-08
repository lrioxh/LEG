import gc
import logging
import os.path as osp
from typing import List, Optional

import evaluate
import numpy as np
import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput

from .trainer import Trainer

logger = logging.getLogger(__name__)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None, labels=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
            "labels": labels.view(-1, 1),
        }

    def __len__(self):
        return self.data["labels"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data
    
    def get_value(self, key):
        return self.data[key]

class InnerTrainer(HugTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):    #TODO：!!重写loss
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if return_outputs:
            logits, hidden_features = model(**inputs, return_hidden=True)
        else:
            logits = model(**inputs, return_hidden=False)   #此处连接E5 forward

        loss_op = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduce="mean")
        loss = loss_op(logits, labels.view(-1))

        if return_outputs:
            outputs = {"logits": logits, "hidden_features": hidden_features}
        return (loss, outputs) if return_outputs else loss


class LMTrainer(Trainer):
    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test", "all"]
        # dataset = TextDataset(self.data.input_ids, self.data.attention_mask, self.data.y)
        return self.data if mode == "all" else torch.utils.data.Subset(self.data, self.split_idx[mode])

    def _prepare_dataset(self):
        return self._get_dataset("train"), self._get_dataset("valid"), self._get_dataset("all")

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = 1* self.args.batch_size_train * self.args.accum_interval
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = int(self.args.warmup_ratio * train_steps)
        training_args = TrainingArguments(
            seed=self.args.seed,
            output_dir=self.args.save,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            # eval_accumulation_steps=10,
            learning_rate=self.args.lm_lr,
            weight_decay=self.args.wd,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            dataloader_drop_last=False,
            gradient_accumulation_steps=self.args.accum_interval,
            label_smoothing_factor=self.args.label_smoothing_factor,
            save_total_limit=1,
            per_device_train_batch_size=self.args.batch_size_train,
            per_device_eval_batch_size=self.args.batch_size_infer,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            disable_tqdm=self.disable_tqdm,
            num_train_epochs=self.args.wu_lm,
            # local_rank=self.rank,
            dataloader_num_workers=8,
            ddp_find_unused_parameters=False,
            deepspeed=self.args.deepspeed,
            fp16=self.args.fp16,
            # bf16=self.args.bf16,
            label_names = ["labels"],
            report_to=self.args.report_to
        )
        return InnerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
