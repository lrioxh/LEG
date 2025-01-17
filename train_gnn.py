#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import gc
import logging
import math
import os
import random
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from functools import lru_cache
from collections import deque
from typing import Literal
import argparse
import json

import optuna
import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

from src.utils import set_logging
from src.misc.revgat.loss import loss_kd_only
from src.model.lm_gnn import RevGAT, GraphSAGE
from src.dataset import load_data_bundle

logger = logging.getLogger(__name__)

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)



def _set_dataset_specific_args(args):
    if args.dataset in ["ogbn-arxiv", "ogbn-arxiv-tape"]:
        args.num_labels = 40
        args.num_feats = 128
        args.expected_valid_acc = 0.6
        args.task_type = "node_cls"

    elif args.dataset == "ogbn-products":
        args.num_labels = 47
        args.num_feats = 100
        args.expected_valid_acc = 0.8
        args.task_type = "node_cls"

    elif args.dataset == "ogbl-citation2":
        args.num_feats = 128
        args.task_type = "link_pred"

    return args

def parse_args():
    parser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--proceed", action="store_true", default=False, help="Continue to train on presaved ckpt")
    parser.add_argument("--suffix", type=str, default="gnn")
    parser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--n_runs", type=int, default=3, help="running times")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")     
    parser.add_argument("--eval_epoch", type=int, default=1) 
    parser.add_argument("--gm_lr", type=float, default=0.002, help="learning rate for GM")
    parser.add_argument("--wd", type=float, default=5e-6, help="weight decay")    
    parser.add_argument("--warmup", type=int, default=10, help="epochs for warmup")    
    parser.add_argument("--loss_reduction", type=str, default='mean', help="Specifies the reduction to apply to the loss output")  
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "constant"],
    )
    parser.add_argument("--label_smoothing_factor", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")   

    # GM
    parser.add_argument("--n_label_iters", type=int, default=2, help="number of label iterations")
    parser.add_argument("--mask_rate", type=float, default=0.5, help="train mask rate")
    parser.add_argument("--no_attn_dst", action="store_true", help="Don't use attn_dst.")
    parser.add_argument("--use_norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--n_heads", type=int, default=2, help="number of heads")
    parser.add_argument("--n_hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--dropout", type=float, default=0.6, help="dropout rate")
    parser.add_argument("--input_drop", type=float, default=0.3, help="input drop rate")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="attention drop rate")
    parser.add_argument("--edge_drop", type=float, default=0.4, help="edge drop rate")
    parser.add_argument("--log_every", type=int, default=1, help="log every LOG_EVERY epochs")
    parser.add_argument("--plot_curves", action="store_true", help="plot learning curves")
    # parser.add_argument("--save_pred", action="store_true", help="save final predictions")
    # parser.add_argument("--save", type=str, default="exp", help="save exp")
    # parser.add_argument("--backbone", type=str, default="rev", help="gcn backbone [deepergcn, wt, deq, rev, gr]")
    parser.add_argument("--group", type=int, default=1, help="num of groups for rev gnns")
    parser.add_argument("--kd_dir", type=str, default="./kd", help="kd path for pred")
    parser.add_argument("--kd_mode", type=str, default="teacher", help="kd mode [teacher, student]")
    parser.add_argument("--alpha", type=float, default=0.5, help="ratio of kd loss")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature of kd")
    
    # parameters for data and model storage
    parser.add_argument("--gnn_type", type=str, default="RevGAT")    # RevGAT GraphSAGE
    parser.add_argument("--se_reduction", type=int, default=16)
    parser.add_argument("--data_folder", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--task_type", type=str, default="node_cls")
    parser.add_argument("--ckpt_dir", type=str, default='', help="path to load gnn ckpt")
    parser.add_argument("--output_dir", type=str, default=f"out")        
    parser.add_argument(
        "--use_labels", action="store_true", default=False, help="Use labels in the training set as input features."
    )
    parser.add_argument("--use_gpt_preds", action="store_true", default=False)
    parser.add_argument("--n_gpt_embs", type=int, default=128)
    parser.add_argument("--use_external_feat", action="store_true", default=False, help="use external static features")
    parser.add_argument("--feat_dir", type=str,default="out/ogbn-arxiv/cached_embs/e5-large-tape-embs.pt", help="path for external static features")
    # out/ogbn-arxiv/cached_embs/e5-large-tape-embs.pt
    # out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt
    parser.add_argument("--train_idx_cluster", action="store_true", default=False)
    # parser.add_argument(
    #     "--ckpt_name", type=str, default="TGRoberta-best.pt"
    # )  # ckpt name to be loaded    
    parser.add_argument(
        "--pretrained_repo",
        type=str,
        help="has to be consistent with repo_id in huggingface",
    )
    
    # dataset and fixed model args
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--num_feats", type=int)
    
    # optuna
    parser.add_argument("--expected_valid_acc", type=float, default=0)
    parser.add_argument("--prune_tolerate", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=18)
    parser.add_argument("--load_study", action="store_true", default=False)
    
    args = parser.parse_args()
    args = _set_dataset_specific_args(args)
    args.save = f"{args.output_dir}/{args.dataset}/{args.gnn_type}/{args.suffix}"
    os.makedirs(f"{args.save}/ckpt",exist_ok=True)
    args.no_attn_dst = True
    args.use_labels = True
    args.use_gpt_preds = True
    args.debug = -1
    # args.proceed = True
    # args.use_external_feat = True
    # args.train_idx_cluster = True
    args.deepspeed = None
    args.disable_tqdm = True
    args.gpt_col = 0
    return args

def save_args(args, dir):
    if int(os.getenv("RANK", -1)) <= 0:
        FILE_NAME = "args.json"
        with open(os.path.join(dir, FILE_NAME), "w") as f:
            json.dump(args.__dict__, f, indent=2)
        logger.info("args saved to {}".format(os.path.join(dir, FILE_NAME)))

def load_args(dir):
    with open(os.path.join(dir, "args.txt"), "r") as f:
        args = argparse.Namespace(**json.load(f))
    return args


class TRAIN_GNN():
    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.epsilon = args.eps if args.eps else 1 - math.log(2)
        self.n_node = 0 
        self.device = None
        self.feat_static = None
        self.gpt_preds = None
        self.graph = None
        self.labels = None
        self.split_idx = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.evaluator = None
        self.optimizer = None
        self.criterion = None
        
        self.model_gnn = None
        
        self.trial = kwargs.pop("trial", None)

    def reorder_train_idx(self):
        '''邻接重排id'''
        visited = set()
        order = []
        train_idx_set = set(self.train_idx.tolist())
        
        # Start BFS from each node in train_idx to ensure all nodes are covered
        for start_node in self.train_idx.tolist():
            if start_node not in visited:
                queue = deque([start_node])
                while queue:
                    node = queue.popleft()
                    if node not in visited and node in train_idx_set:
                        visited.add(node)
                        order.append(node)
                        neighbors = self.graph.successors(node).tolist()
                        queue.extend(neighbors)
        
        self.train_idx = torch.tensor(order)

    
    def custom_train_loss(self, labels, x1, x2 = None):
        y1 = self.criterion(x1, labels[:, 0])
        y = torch.log(self.epsilon + y1) - math.log(self.epsilon)
        # if x2 != None: 
        #     y2 = self.criterion(x2, labels[:, 0])
        #     y += torch.log(self.epsilon + y2) - math.log(self.epsilon)
        return torch.mean(y)
    
    def custom_eval_loss(self, labels, x1, x2 = None, label_smoothing_factor = 0):
        # 与train_loss一样，实现方法不同
        y = F.cross_entropy(x1, labels[:, 0], reduction=self.args.loss_reduction, label_smoothing=label_smoothing_factor)
        y = torch.log(self.epsilon + y) - math.log(self.epsilon)
        # if x2 != None: 
        #     y2 = F.cross_entropy(x2, labels[:, 0], reduction="none", label_smoothing=label_smoothing_factor)
        #     y += torch.log(self.epsilon + y2) - math.log(self.epsilon)
        return torch.mean(y)
    
    def cal_labels(self, length, labels, idx):
        '''label编码'''
        onehot = torch.zeros([length, self.args.num_labels], device=self.device, 
                            #  dtype=torch.float16 if self.args.fp16 else torch.float32
                            )
        if len(idx)>0:
            onehot[idx, labels[idx, 0]] = 1
        return onehot
    
    def preprocess(self):
        # global n_node_feats

        # make bidirected
        # feat = graph.ndata["feat"]
        self.graph = dgl.to_bidirected(self.graph)
        self.graph.ndata["feat"] = torch.empty((self.graph.num_nodes(), 0))

        # add self-loop
        logger.info(f"Total edges before adding self-loop {self.graph.number_of_edges()}")
        self.graph = self.graph.remove_self_loop().add_self_loop()
        logger.info(f"Total edges after adding self-loop {self.graph.number_of_edges()}")

        self.graph.create_formats_()
    
    def prepare(self):
        '''device, scaler, criterion'''
        if self.args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.args.gpu}")

        self.labels = self.labels.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduction =self.args.loss_reduction)

    def adjust_learning_rate(self, epoch):
        '''lr schedule'''
        # if not self.lm_only:
        if epoch <= self.args.warmup:     #TODO: 分层调整
        #     lm_lr = self.args.lm_lr * epoch / self.args.warmup
            # gm_lr = self.args.gm_lr * 0.5*epoch*(1 + 1/self.args.warmup) 
            gm_lr = self.args.gm_lr * epoch / self.args.warmup
            # for i, param_group in enumerate(self.optimizer.param_groups):
            #     param_group["lr"] = gm_lr
        else:
            # dec = 4 * np.exp(-0.1 * (epoch - self.args.warmup + 14))
            dec = (1 - (epoch-self.args.warmup) / (self.args.n_epochs*2))
            # dec = (1 - (epoch-self.args.warmup) / self.args.n_epochs)
            # dec = 1
            gm_lr = self.args.gm_lr * dec
        # lm_lr = self.optimizer.param_groups[0]["lr"]
        # gm_lr = self.optimizer.param_groups[-1]["lr"]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = gm_lr
        logger.info(f"gm_lr: {gm_lr}")

    def to_device(self, item):
        if item != None:
            item.to(self.device)
            
    def save_pred(self, pred, name):
        os.makedirs(f"{self.args.save}/cached_embs", exist_ok=True)
        torch.save(pred, f"{self.args.save}/cached_embs/logits_{name}.pt")
        torch.save(self.feat_static, f"{self.args.save}/cached_embs/x_embs_{name}.pt")
        logger.warning(f"Saving logits & x_embs to {self.args.save}/cached_embs/_{name}.pt")
        
    def save_model(self, run_num, epoch):
        out_dir = f"{self.args.save}/ckpt"
        os.makedirs(out_dir,exist_ok=True)
        fname_gnn = os.path.join(out_dir, f"{epoch}_run_{run_num}_gnn.pt")
        torch.save(self.model_gnn.state_dict(), fname_gnn)  
    
    def get_params(self, init_lr=False, need_name = False, grad_only = True):
        params = []
        if init_lr:
            if self.model_gnn:
                gmp = [{'params': p, 'lr': self.args.gm_lr} for p in self.model_gnn.parameters()]
                params += gmp
            if grad_only:
                return [p for p in params if p['params'].requires_grad]
        elif need_name:
            if self.model_gnn:
                params += list(self.model_gnn.named_parameters())
            if grad_only:
                return [(n,p) for (n,p) in params if p.requires_grad]
        else:
            if self.model_gnn:
                params += list(self.model_gnn.parameters())
            if grad_only:
                return [p for p in params if p.requires_grad]
        return params
            
      
    def count_params(self, grad_only=True):
        params = self.get_params(grad_only = grad_only)
        return sum([p.numel() for p in params])
    
    def print_grad_norm(self):
        for name, param in self.get_params():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                logger.info(f"Layer: {name} | Gradient Norm: {grad_norm}")
       
    def load_data(self):
        assert self.args.dataset in [
                "ogbn-arxiv", "ogbl-citation2", "ogbn-products", "ogbn-arxiv-tape"
            ]
        data_graph = DglNodePropPredDataset(name=self.args.dataset, root=self.args.data_folder)
        self.evaluator = Evaluator(name=self.args.dataset)
        
        self.split_idx = data_graph.get_idx_split()
        self.train_idx, self.val_idx, self.test_idx = self.split_idx ["train"], self.split_idx ["valid"], self.split_idx ["test"]
        self.graph, self.labels = data_graph[0]
        
        # self.args.n_node_feats = self.args.hidden_size
        
        if self.args.debug > 0:
            if self.args.dataset=='ogbn-arxiv':
                debug_idx = torch.arange(0, self.args.debug)
                self.train_idx = self.train_idx[self.train_idx < self.args.debug]
                self.val_idx = self.val_idx[self.val_idx < self.args.debug]
                self.test_idx = self.test_idx[self.test_idx < self.args.debug]
                self.labels = self.labels[:self.args.debug]
            elif self.args.dataset=='ogbn-products':
                data_ = torch.load(f'{self.args.data_folder}/ogbn_products_subset.pt')
                debug_idx = data_.n_id
                new_idx = torch.arange(0, len(debug_idx))
                self.train_idx = new_idx[data_.train_mask]
                self.val_idx = new_idx[data_.val_mask]
                self.test_idx = new_idx[data_.test_mask]
                self.labels = self.labels[debug_idx]
                
            self.split_idx["train"] = self.train_idx
            self.split_idx["valid"] = self.val_idx
            self.split_idx["test"] = self.test_idx
            self.graph = dgl.node_subgraph(self.graph, debug_idx)
        
        if self.args.use_external_feat:
            self.feat_static = torch.load(self.args.feat_dir)
            self.args.n_node_feats = self.feat_static.shape[1]
            logger.warning(
                f"Loaded node embeddings of shape={self.feat_static.shape} from {self.args.feat_dir}"
            )
        elif self.args.use_gpt_preds:
            preds = []
            with open(f"src/misc/gpt_preds/{self.args.dataset}.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    preds.append([int(i) for i in row])
                    self.args.gpt_col = max(self.args.gpt_col, len(row))
            pl = torch.zeros(len(preds), self.args.gpt_col, dtype=torch.long)
            for i, pred in enumerate(preds):
                pl[i][: len(pred)] = torch.tensor(pred[:self.args.gpt_col], dtype=torch.long) + 1
            self.feat_static = pl
            logger.warning(
                "Loaded node embeddings of shape={} from gpt_preds".format(pl.shape)
            )
            
            self.args.n_node_feats = self.args.n_gpt_embs * self.args.gpt_col
        else:
            self.feat_static = self.graph.ndata["feat"]
            self.args.n_node_feats = self.feat_static.shape[1]
            logger.warning(
                "Use node embeddings of shape={} from ogb".format(pl.shape)
            )
            
        self.n_node = self.graph.num_nodes()
        if self.args.use_labels:
            self.args.n_node_feats += self.args.num_labels
        if self.args.train_idx_cluster:
            self.reorder_train_idx()
        return 1

    def gen_model(self):
        if self.args.gnn_type == "RevGAT":
            self.model_gnn = RevGAT(
                self.args,
                activation=F.relu,
                gpt_col=self.args.gpt_col,
                dropout=self.args.dropout,
                input_drop=self.args.input_drop,
                attn_drop=self.args.attn_drop,
                edge_drop=self.args.edge_drop,
                use_attn_dst=not self.args.no_attn_dst,
                use_symmetric_norm=self.args.use_norm,
                use_gpt_preds=self.args.use_gpt_preds,
                se_reduction=self.args.se_reduction
            )

            if self.args.ckpt_dir != '' and os.path.exists(self.args.ckpt_dir):
                self.model_gnn.load_state_dict(torch.load(self.args.ckpt_dir),strict=False)
                logger.info(f"Loaded PGM from {self.args.ckpt_dir}")
                self.model_gnn.convs[-1].reset_parameters()
        elif self.args.gnn_type == "GraphSAGE":
            # print(self.args.n_node_feats)
            self.model_gnn = GraphSAGE(
                in_channels=self.args.n_node_feats,
                hidden_channels=self.args.n_hidden,
                out_channels=self.args.num_labels,
                num_layers=self.args.n_layers,
                dropout=self.args.dropout,
                use_gpt_preds=self.args.use_gpt_preds
            )
        else:
            raise Exception(f"Unknown gnn {self.args.gnn_type}")
            
        self.optimizer = optim.RMSprop(self.get_params(init_lr=True), lr=self.args.gm_lr, weight_decay=self.args.wd)
        return 1

    def train(
        self, epoch, evaluator, mode="teacher", teacher_output=None
    ):
        
        self.model_gnn.train()
        
        # if mode == "student":
        #     assert teacher_output != None

        alpha = self.args.alpha
        temp = self.args.temp

        
        feat_train = self.feat_static.to(device=self.device)
        graph = self.graph.to(device=self.device)
            
        if self.args.use_labels:
            feat_train = torch.cat([feat_train,       
                              torch.zeros((self.n_node, self.args.num_labels), 
                                          device=self.device)],
                              dim=-1)

        self.optimizer.zero_grad()
        if self.args.use_labels:
            mask = torch.rand(self.train_idx.shape) < self.args.mask_rate
            train_labels_idx = self.train_idx[mask]
            train_pred_idx = self.train_idx[~mask]
        else:
            mask = torch.rand(self.train_idx.shape) < self.args.mask_rate
            train_pred_idx = self.train_idx[mask]

        if self.args.n_label_iters > 0:
            with torch.no_grad():
                pred = self.model_gnn(graph, feat_train)
        else:
            pred = self.model_gnn(graph, feat_train)

        if self.args.n_label_iters > 0:
            unlabel_idx = torch.cat([train_pred_idx, self.val_idx, self.test_idx])
            for _ in range(self.args.n_label_iters):
                pred = pred.detach()
                torch.cuda.empty_cache()
                feat_train[unlabel_idx, -self.args.num_labels:] = F.softmax(pred[unlabel_idx], dim=-1)
                pred = self.model_gnn(graph, feat_train)

        if mode == "teacher":
            loss = self.custom_train_loss(self.labels[train_pred_idx], pred[train_pred_idx])
        # elif mode == "student":
        #     loss_gt = self.custom_train_loss(self.labels[train_pred_idx], pred[train_pred_idx])
        #     loss_kd = loss_kd_only(pred, teacher_output, temp)
        #     loss = loss_gt * (1 - alpha) + loss_kd * alpha
        else:
            raise Exception("unkown mode")

        # if self.args.fp16:
        #     self.scaler.scale(loss).backward()
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        # else:
        loss.backward()
        self.optimizer.step()

        return evaluator(pred[self.train_idx], self.labels[self.train_idx]), loss.item()


    @torch.no_grad()
    def evaluate(self, evaluator):

        self.model_gnn.eval()

        # feat = graph.ndata["feat"]
        graph = self.graph.to(device=self.device)
        feat_eval = self.feat_static.to(self.device)
            
        if self.args.use_labels:
            onehot_labels = self.cal_labels(self.n_node, self.labels, self.train_idx)
            feat_eval = torch.cat([feat_eval, onehot_labels], dim=-1)
            
        pred = self.model_gnn(graph, feat_eval)

        if self.args.n_label_iters > 0:
            unlabel_idx = torch.cat([self.val_idx, self.test_idx])
            for _ in range(self.args.n_label_iters):
                onehot_labels[unlabel_idx] = F.softmax(pred[unlabel_idx], dim=-1)
                pred = self.model_gnn(graph, feat_eval)
        #TODO: eval也计算lmloss
        train_loss = self.custom_eval_loss(self.labels[self.train_idx], pred[self.train_idx])
        val_loss = self.custom_eval_loss(self.labels[self.val_idx], pred[self.val_idx])
        test_loss = self.custom_eval_loss(self.labels[self.test_idx], pred[self.test_idx])

        return (
            evaluator(pred[self.train_idx], self.labels[self.train_idx]),
            evaluator(pred[self.val_idx], self.labels[self.val_idx]),
            evaluator(pred[self.test_idx], self.labels[self.test_idx]),
            train_loss,
            val_loss,
            test_loss,
            pred,
        )


    def run(self, n_running, rseed, prune_tolerate = 1):
        evaluator_wrapper = lambda pred, labels: self.evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels} #onehot to cls
        )["acc"]

        # kd mode
        mode = self.args.kd_mode
        # define model and optimizer
        #e5_revgat
        self.gen_model()
        start_ep = 0
        # if self.args.proceed:
        #     start_ep, last_is_full_ft = self.load_stat()
        
        logger.info(f"Number of all params: {self.count_params(grad_only=False)}")
        self.to_device(self.model_gnn)
        # training loop
        total_time = 0
        best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
        final_pred = None

        accs, train_accs, val_accs, test_accs = [], [], [], []
        losses, train_losses, val_losses, test_losses = [], [], [], []
        epoch = start_ep + 1
        
        while epoch < self.args.n_epochs + 1:
        # for epoch in range(start_ep + 1, self.args.n_epochs + 1):
            tic = time.time()         
            # if mode == "student":
            #     teacher_output = torch.load("./{}/best_pred_run{}.pt".format(self.args.kd_dir, n_running)).cpu().cuda()
            # else:
            #     teacher_output = None

            # if self.lm_only:
            #     train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = self.train_lm()
            #     pred = None
            # else:
                        
            self.adjust_learning_rate(epoch)
            
            acc, loss = self.train(
                epoch,
                evaluator_wrapper,
                mode=mode,
                # teacher_output=teacher_output,
            )

            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = \
                                                                self.evaluate(evaluator_wrapper)
            if self.trial and prune_tolerate == 0:
                if val_acc < self.args.expected_valid_acc or self.trial.should_prune():
                    logger.critical(
                        f"valid acc {val_acc:.4f} is lower than expected {self.args.expected_valid_acc:.4f}"
                    )
                    raise optuna.exceptions.TrialPruned()
                
            toc = time.time()
            total_time += toc - tic

            # if epoch == 1:
            peak_memuse = torch.cuda.max_memory_allocated(self.device) / float(1024**3)
            logger.info("Peak memuse {:.2f} G".format(peak_memuse))

            if val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_acc = test_acc
                final_pred = pred
                # if mode == "teacher":
                #     self.save_pred(final_pred, n_running, self.args.kd_dir)
                if val_acc > 0.7:
                    self.save_pred(final_pred, f'best_{rseed}')
                    logger.info(f'best{rseed} at ep{epoch} saved')

            if epoch == self.args.n_epochs or epoch % self.args.log_every == 0:
                logger.info(
                    f"Run: {n_running}/{self.args.n_runs}/{rseed}, Epoch: {epoch}/{self.args.n_epochs}, Average epoch time: {total_time / (epoch-start_ep):.2f}\n"
                    f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
                )

            for l, e in zip(
                [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
                [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)
            
            epoch+=1

        logger.info("*" * 50)
        logger.info(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
        logger.info("*" * 50)

        return best_val_acc, final_test_acc



def main():
    set_logging()
    gbc = TRAIN_GNN(parse_args())
    
    if not gbc.args.use_labels and gbc.args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    # load data & preprocess
    gbc.load_data()
    gbc.preprocess()#

    # to device
    gbc.prepare()
    logger.info(gbc.args)
    save_args(gbc.args, gbc.args.save)
    
    # run
    val_accs, test_accs = [], []

    for i in range(gbc.args.n_runs):
        rseed = gbc.args.seed + i
        seed(rseed)
        val_acc, test_acc = gbc.run(i + 1, rseed, gbc.args.prune_tolerate)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    logger.info(gbc.args)
    logger.info(f"Runned {gbc.args.n_runs} times")
    logger.info("Val Accs:")
    logger.info(val_accs)
    logger.info("Test Accs:")
    logger.info(test_accs)
    logger.info(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    logger.info(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    logger.info(f"Number of all params: {gbc.count_params(grad_only=False)}")
    logger.info(f"Number of trainable params: {gbc.count_params()}")


def count_params():
    gbc = TRAIN_GNN(parse_args())
    gbc.load_data()
    gbc.gen_model()
    print(f"Params ALL: {gbc.count_params()}")
    print(f"Params trainable: {gbc.count_params()}")

if __name__ == "__main__":
    main()
    # count_params()
