import gc
import logging
import os
import os.path as osp
import evaluate
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
# import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist
import torch.distributed as dist

# from .trainer import Trainer

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
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


class Trainer(ABC):
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.iter = 0
        self.trial = kwargs.get("trial", None)

    @property
    def rank(self):
        return int(os.environ["RANK"]) if is_dist() else -1

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"]) if is_dist() else 1

    @property
    def disable_tqdm(self):
        return self.args.disable_tqdm or (is_dist() and self.rank > 0)

    @property
    def ckpt_path(self):
        return osp.join(self.args.ckpt_dir, "model.pt")

    def save_model(self, model: torch.nn.Module, ckpt_path):
        if self.rank <= 0:
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved the model to {}".format(ckpt_path))
        if is_dist():
            dist.barrier()

    def load_model(self, model: torch.nn.Module, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    def _prepare_model(self):
        model_class = get_model_class(self.args.model_type, self.args.task_type)
        model = model_class(self.args)
        n_params = sum(p.numel() for p in model.parameters())
        logger.warning(f"Model: {self.args.model_type}, Num of Params: {n_params}")
        return model

    @abstractmethod
    def _prepare_dataset(self):
        pass

    @abstractmethod
    def _prepare_trainer(self):
        pass

    def compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(-1)
        return metric.compute(predictions=predictions, references=labels)

    def inference(self, dataset, embs_path):
        x_embs_name = f"x_embs.pt"
        logits_name = f"logits.pt"
        emb_handler = EmbeddingHandler(embs_path)
        if self.args.use_cache and emb_handler.has([x_embs_name, logits_name]):
            x_embs = emb_handler.load(x_embs_name)
            logits_embs = emb_handler.load(logits_name)
            if isinstance(x_embs, np.ndarray):
                x_embs, logits_embs = torch.from_numpy(x_embs), torch.from_numpy(logits_embs)
        else:
            eval_output = self.trainer.predict(dataset)
            logits_embs, x_embs = eval_output.predictions[0], eval_output.predictions[1]
            logits_embs, x_embs = torch.from_numpy(logits_embs), torch.from_numpy(x_embs)
            emb_handler.save(x_embs, x_embs_name)   #save embs
            emb_handler.save(logits_embs, logits_name)
            logger.info(f"save the logits of {self.args.lm_type} to {osp.join(embs_path, logits_name)}")
            logger.info(f"save the hidden features of {self.args.lm_type} to {osp.join(embs_path, x_embs_name)}")
        return logits_embs, x_embs

    def _evaluate(self, logits, y):
        def accuracy(logits, y_true):
            y_pred = logits.argmax(dim=-1, keepdim=True)
            acc = y_pred.eq(y_true.view_as(y_pred)).sum() / y_true.shape[0]
            return acc.item()

        results = dict()
        for split in ["train", "valid", "test"]:
            split_idx = self.split_idx[split]
            acc = accuracy(logits[split_idx], y[split_idx])
            results[f"{split}_acc"] = acc
            if logits.dtype is not torch.half:
                loss = F.cross_entropy(logits[split_idx], y[split_idx].view(-1)).item()
                results[f"{split}_loss"] = loss
        return results

    def inference_and_evaluate(self, dataset):
        embs_path = os.path.join(self.args.output_dir, "cached_embs")
        logits_embs, x_embs = self.inference(dataset, embs_path)    #save embs
        results = self._evaluate(logits_embs, self.data.y)
        logger.critical("".join("{}:{:.4f} ".format(k, v) for k, v in results.items()))
        gc.collect()
        torch.cuda.empty_cache()
        return logits_embs, x_embs, results  # x_embs is None in GNNTrainer

    def train_once(self):
        if is_dist(): dist.barrier()
        if self.trial is not None:
            self.trainer._hp_search_setup(self.trial)
        train_output = self.trainer.train()
        # save model
        self.save_model(self.model, self.ckpt_path)
        global_step, train_dict = train_output.global_step, train_output.metrics
        train_dict["global_step"] = global_step
        self.trainer.save_metrics("train", train_dict)
        logger.critical("".join("{}:{} ".format(k, v) for k, v in train_dict.items()))
        gc.collect()
        torch.cuda.empty_cache()

    def prepare(self):
        self.model = self._prepare_model()
        self.train_set, self.valid_set, self.all_set = self._prepare_dataset()
        self.trainer = self._prepare_trainer()

    def train(self, return_value="valid"):
        self.prepare()
        assert self.args.mode in ["train", "test"]
        if self.args.mode == "train":
            self.train_once()

        logger.warning(f"\n*************** Start inference and testing ***************\n")
        _, _, results = self.inference_and_evaluate(self.all_set)
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(self.model.state_dict(), self.ckpt_path)
        return results["test_acc"], results["valid_acc"]


class InnerTrainer(HugTrainer): #单步
    def compute_loss(self, model, inputs, return_outputs=True):    #TODO：!!重写loss
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


class LM_GNN_Trainer(Trainer):
    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test", "all"]
        dataset = Dataset(self.data.input_ids, self.data.attention_mask, self.data.y)
        return dataset if mode == "all" else torch.utils.data.Subset(dataset, self.split_idx[mode])

    def _prepare_dataset(self):
        return self._get_dataset("train"), self._get_dataset("valid"), self._get_dataset("all")

    def _prepare_trainer(self): #重写
        # prepare training args
        total_batch_size = 1 * self.args.batch_size * self.args.accum_interval
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = self.args.warmup_ratio * train_steps
        training_args = TrainingArguments(
            seed=self.args.random_seed,
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            # eval_accumulation_steps=10,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            dataloader_drop_last=True,
            gradient_accumulation_steps=self.args.accum_interval,
            label_smoothing_factor=self.args.label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            disable_tqdm=False,
            num_train_epochs=self.args.epochs,
            local_rank=self.rank,
            dataloader_num_workers=8,
            ddp_find_unused_parameters=False,
            deepspeed=self.args.deepspeed,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
        )
        return InnerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
