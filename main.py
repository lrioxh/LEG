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
from src.model.lm_gnn import RevGAT, E5_model
from src.dataset import load_data_bundle
from src.args import parse_args, save_args
import src.lora as lora

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


def preprocess(graph):
    # global n_node_feats

    # make bidirected
    # feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = torch.empty((graph.num_nodes(), 0))

    # add self-loop
    logger.info(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    logger.info(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph

class ReplaceRowsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rows, replacement):
        ctx.save_for_backward(input, rows, replacement)
        output = input.clone()
        output[rows] = replacement
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, rows, replacement = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_replacement = grad_output[rows].clone()
        grad_input[rows] = 0
        return grad_input, None, grad_replacement
    
replace_rows = ReplaceRowsFunction.apply

        
class LM_GNN():
    def __init__(self, args) -> None:
        self.args = args
        self.epsilon = 1 - math.log(2)
        # dataset = "ogbn-arxiv"
        self.n_node = 0 
        self.n_classes = 0
        self.device = None
        self.text_data = None
        self.feat_static = None
        self.gpt_preds = None
        self.graph = None
        self.graph_loader = None
        self.labels = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.evaluator = None
        self.optimizer = None
        self.criterion = None
        self.whole_graph = False
        
        self.model_lm = None
        self.model_gnn = None
        self.is_lm = []
        self.require_grad = []

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
        onehot = torch.zeros([length, self.n_classes], device=self.device, 
                            #  dtype=torch.float16 if self.args.fp16 else torch.float32
                            )
        if len(idx)>0:
            onehot[idx, labels[idx, 0]] = 1
        return onehot
    
    def prepare(self):
        '''device, scaler, criterion'''
        if self.args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.args.gpu}")
        # self.labels, self.val_idx, self.test_idx = map(
        # lambda x: x.to(self.device), (self.labels, self.val_idx, self.test_idx)
        # )
        self.labels = self.labels.to(self.device)
        # 初始化GradScaler
        self.scaler = GradScaler() if self.args.fp16 else None
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduction =self.args.loss_reduction)

    def adjust_learning_rate(self, epoch, full_ft):
        '''lr schedule'''
        if epoch <= self.args.warmup:     #TODO: 分层调整
            lm_lr = self.args.lm_lr * epoch / self.args.warmup
            gm_lr = self.args.gm_lr * epoch / self.args.warmup
            for i, param_group in enumerate(self.optimizer.param_groups):
                if self.is_lm[i]:
                    param_group["lr"] = lm_lr
                else:
                    param_group["lr"] = gm_lr
        else:
            lm_lr = self.optimizer.param_groups[0]["lr"]
            gm_lr = self.optimizer.param_groups[-1]["lr"]
        logger.info(f"lm_lr: {lm_lr}, gm_lr: {gm_lr}")

    def to_device(self, item):
        if item != None:
            item.to(self.device)
            
    def save_pred(self, pred, run_num, kd_dir):
        os.makedirs(kd_dir,exist_ok=True)
        fname = os.path.join(kd_dir, "best_pred_run{}.pt".format(run_num))
        torch.save(pred.cpu(), fname)  
        
    def save_model(self, run_num, epoch):
        out_dir = f"{self.args.save}/ckpt"
        os.makedirs(out_dir,exist_ok=True)
        fname_gnn = os.path.join(out_dir, f"{epoch}_run_{run_num}_gnn.pt")
        torch.save(self.model_gnn.state_dict(), fname_gnn)  
        if self.model_lm:
            fname_lm = os.path.join(out_dir, f"{epoch}_run_{run_num}_lm.pt")
            torch.save(self.model_lm.state_dict(), fname_lm)  
        
    def save_stat(self, epoch, full_ft, name):
        out_dir = f"{self.args.save}/ckpt"
        fname = os.path.join(out_dir, f"{name}_stat.pt")
        torch.save({
            'epoch': epoch,
            'gnn_dict': self.model_gnn.state_dict(),
            'lm_dict': self.model_lm.state_dict() if self.model_lm else None,
            'optm_dict': self.optimizer.state_dict(),
            'feat_static': self.feat_static,
            'full_ft': full_ft,
            # 可以添加其他你需要保存的状态
        }, fname)
        logger.info(f"Saving stat ckpt for ep{epoch} ...")
    
    def load_stat(self):
        out_dir = f"{self.args.save}/ckpt"
        fname = os.path.join(out_dir, f"last_stat.pt")
        checkpoint = torch.load(fname, map_location=self.device)
        last_epoch = checkpoint['epoch']  # 从上次结束的epoch开始
        full_ft = checkpoint['full_ft']  # 从上次结束的epoch开始
        logger.info(f"Loading last ckpt from {fname}, continue after ep{last_epoch}")
        if 0 < self.args.peft_start <= last_epoch:
            self.switch_to('gnn_lora')     
        # else:
            # self.optimizer = optim.RMSprop(self.get_params(), lr=self.args.gm_lr, weight_decay=self.args.wd)
        self.to_device(self.model_gnn)
        self.to_device(self.model_lm)
        self.model_gnn.load_state_dict(checkpoint['gnn_dict'],strict=False)
        if self.model_lm: self.model_lm.load_state_dict(checkpoint['lm_dict'])
        # if self.args.peft_start != last_epoch: 
        self.optimizer.load_state_dict(checkpoint['optm_dict'])
        self.feat_static = checkpoint['feat_static']
        del checkpoint
        return last_epoch, full_ft
    
    def get_params(self, grad_only = True):
        params = []
        if self.model_lm:
            lmp = [{'params': p, 'lr': self.args.lm_lr} for p in self.model_lm.parameters()]
            params += lmp
            self.is_lm += [1 for _ in range(len(lmp))]
        if self.model_gnn:
            gmp = [{'params': p, 'lr': self.args.gm_lr} for p in self.model_gnn.parameters()]
            params += gmp
            self.is_lm += [0 for _ in range(len(gmp))]
        if grad_only:
            return [p for p in params if p['params'].requires_grad]
        else:
            return params
      
    def count_params(self, grad_only=True):
        return sum([p['params'].numel() for p in self.get_params(grad_only)])
    
    @lru_cache(8)
    def id_in_parent(self, parent, sub):
        '''index transformation in subset'''
        if self.whole_graph:
            return sub
        sorted_parent, sorted_indices = torch.sort(parent)
        sorted_pos = torch.searchsorted(sorted_parent, sub)
        return sorted_indices[sorted_pos]

    # @torch.no_grad()
    def get_feat(self, device='cpu', return_cls = False):
        text_loader = DataLoader(self.text_data, batch_size=self.args.batch_size, shuffle=False)
        num_batches = len(text_loader)
        interval = num_batches//10
        feat = torch.empty((self.n_node, self.args.hidden_size), 
                                dtype=torch.float16 if self.args.fp16 else torch.float32, device=self.device, requires_grad=False)
        out = torch.empty((self.n_node, self.n_classes),
                        dtype=torch.float16 if self.args.fp16 else torch.float32, device=self.device)
        with tqdm(total=num_batches, desc=f'LM inference', unit='batch', file=open(os.devnull, 'w')) as pbar:
        #     with logging_redirect_tqdm():
        # pbar = tqdm(range(num_batches), file=open(os.devnull, 'w'))
            for i, (input_ids, attention_mask) in enumerate(text_loader):
                # lm
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if self.args.fp16:
                    with autocast():    
                        out[i*self.args.batch_size:i*self.args.batch_size+input_ids.shape[0]], \
                        feat[i*self.args.batch_size:i*self.args.batch_size+input_ids.shape[0],:self.args.hidden_size] \
                                        = self.model_lm(input_ids,attention_mask,return_hidden=True)
                else:
                    out[i*self.args.batch_size:i*self.args.batch_size+input_ids.shape[0]], \
                    feat[i*self.args.batch_size:i*self.args.batch_size+input_ids.size[0],:self.args.hidden_size] \
                                        = self.model_lm(input_ids,attention_mask,return_hidden=True)
                
                if interval == 0: logger.info(str(pbar))
                elif (i-1) % interval == 0: logger.info(str(pbar))
                pbar.update(1)
            torch.cuda.empty_cache()
            gc.collect()
        if return_cls:
            return out, feat.to(device)
        else:
            return feat.to(device)
         
    def load_data(self):
        assert self.args.dataset in [
                "ogbn-arxiv", "ogbl-citation2", "ogbn-products", "ogbn-arxiv-tape"
            ]
        data_graph = DglNodePropPredDataset(name=self.args.dataset, root="../dgl_data")
        self.evaluator = Evaluator(name=self.args.dataset)
        
        splitted_idx = data_graph.get_idx_split()
        self.train_idx, self.val_idx, self.test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        self.graph, self.labels = data_graph[0]
        
        self.n_node = self.graph.num_nodes()
        self.n_classes = (self.labels.max() + 1).item()

        if self.args.use_external_feat:
            self.feat_static = torch.load(self.args.feat_dir)
            logger.warning(
                f"Loaded pre-trained node embeddings of shape={self.feat_static.shape} from {self.args.feat_dir}"
            )
        else:
            # text attr
            text_token, split_idx, evaluator = load_data_bundle(
                self.args.dataset,
                root=self.args.data_folder,
                tokenizer=self.args.pretrained_repo,
                tokenize=True)
            if self.args.dataset == "ogbn-arxiv":
                transform = T.ToUndirected()    #TODO: 加入PE处理有向图
                text_token = transform(text_token)
            self.text_data = TensorDataset(text_token.input_ids, text_token.attention_mask) 
            logger.warning(
                f"Loaded node tokens of shape=({self.n_node},{text_token.input_ids.shape[1]})")      
        # TODO
        self.args.n_node_feats = self.args.hidden_size
        if self.args.use_gpt_preds:
            preds = []
            with open(f"src/misc/gpt_preds/ogbn-arxiv.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    preds.append([int(i) for i in row])
            pl = torch.zeros(len(preds), 5, dtype=torch.long)
            for i, pred in enumerate(preds):
                pl[i][: len(pred)] = torch.tensor(pred[:5], dtype=torch.long) + 1
            self.gpt_preds = pl
            logger.warning(
                "Loaded pre-trained node embeddings of shape={} from gpt_preds".format(pl.shape)
            )
            
            self.args.n_node_feats += self.args.n_gpt_embs * 5
        
        if self.args.debug > 0:
            debug_idx = [i for i in range(self.args.debug)]
            self.train_idx = self.train_idx[self.train_idx < self.args.debug]
            self.val_idx = self.val_idx[self.val_idx < self.args.debug]
            self.test_idx = self.test_idx[self.test_idx < self.args.debug]
            self.labels = self.labels[:self.args.debug]
            self.graph = dgl.node_subgraph(self.graph, debug_idx)
            self.text_data = Subset(self.text_data, debug_idx)

        if self.args.use_labels:
            self.args.n_node_feats += self.n_classes
        if self.args.train_idx_cluster:
            self.reorder_train_idx()
        return 1

    def init_loader(self):
        '''dataloader, sampling here'''
        if self.args.grad_padding > 0:
            # 控制采样：低->高，1随机一个节点，-1所有节点
            grad_block = [self.args.grad_k for _ in range(self.args.grad_padding)]
            if self.args.frozen_padding >= 0: 
                fz_block = [1]+[-1 for _ in range(self.args.frozen_padding)]
                sampler = dgl.dataloading.NeighborSampler(fz_block + grad_block)
                self.graph_loader = dgl.dataloading.DataLoader(
                    self.graph, self.train_idx, sampler,
                    batch_size=self.args.kernel_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=4)
            else:
                sampler = dgl.dataloading.ShaDowKHopSampler(grad_block)
                self.graph_loader = dgl.dataloading.DataLoader(
                    self.graph, self.train_idx, sampler,
                    batch_size=self.args.kernel_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=4)
        else:
            # TODO: whole graph
            self.whole_graph = True

    def gen_model(self):

        if self.args.gnn_type == "RevGAT":
            self.model_gnn = RevGAT(
                self.args,
                self.n_classes,
                activation=F.relu,
                dropout=self.args.dropout,
                input_drop=self.args.input_drop,
                attn_drop=self.args.attn_drop,
                edge_drop=self.args.edge_drop,
                use_attn_dst=not self.args.no_attn_dst,
                use_symmetric_norm=self.args.use_norm,
                use_gpt_preds=self.args.use_gpt_preds,
                lora_params={
                    'use_lora': self.args.use_peft,
                    'r': self.args.peft_r,
                    'lora_alpha': self.args.peft_lora_alpha,
                    'lora_dropout': self.args.peft_lora_dropout
                    } if 0 < self.args.peft_start <= self.args.n_epochs else None
            )

            if self.args.ckpt_dir != '' and os.path.exists(self.args.ckpt_dir):
                self.model_gnn.load_state_dict(torch.load(self.args.ckpt_dir),strict=False)
                logger.info(f"Loaded PGM from {self.args.ckpt_dir}")
                self.model_gnn.convs[-1].reset_parameters()
        else:
            raise Exception("Unknown gnn")
        if not self.args.use_external_feat:
            if self.args.lm_type == "e5-large":
                self.model_lm = E5_model(self.args)
            else:
                raise Exception("Unknown lm")
            
        self.optimizer = optim.RMSprop(self.get_params(), lr=self.args.gm_lr, weight_decay=self.args.wd)
        self.require_grad = [0 for _ in self.model_lm.parameters()]
        return 1

    def switch_to(self, mode: Literal['gnn_lora', 'gnn_backbone', 'gnn_only', 'full_ft']):
        if mode=='gnn_lora':
            lora.mark_only_lora_as_trainable(self.model_gnn)
            logger.info("GM switched to LoRA")
        elif mode == 'gnn_backbone':
            lora.mark_not_lora_as_trainable(self.model_gnn)
            logger.info("GM switched to backbone")
        elif mode == 'gnn_only':
            for i, p in enumerate(self.model_lm.parameters()):
                if p.requires_grad == True:
                    p.requires_grad = False
                    self.require_grad[i] = 1
            logger.info("Switched to gnn only")
        elif mode == 'full_ft':
            for i, p in enumerate(self.model_lm.parameters()):
                if self.require_grad[i]:
                    p.requires_grad = True
            logger.info("Switched to lm+gnn")
        else:
            logger.info(f"Invalid switch mode: {mode}")
        logger.info(f"Number of trainable params: {self.count_params()}")

    def train(
        self, epoch, evaluator, full_ft, mode="teacher", teacher_output=None
    ):
        self.model_gnn.train()
        if self.model_lm: self.model_lm.train()
        
        if mode == "student":
            assert teacher_output != None

        alpha = self.args.alpha
        temp = self.args.temp

        if full_ft and self.whole_graph:
            out_lm, feat_train = self.get_feat(self.device, return_cls = True)
        else:
            if self.feat_static == None:
                with torch.no_grad():
                    self.feat_static = self.get_feat()    
            feat_train = self.feat_static.to(self.device)
            
        if self.args.use_labels:
            feat_train = torch.cat([feat_train, 
                              torch.zeros((self.n_node, self.n_classes), 
                                          dtype=torch.float16 if self.args.fp16 else torch.float32, 
                                          device=self.device)],
                              dim=-1)
        if self.args.use_gpt_preds:
            feat_train = torch.cat([self.gpt_preds.to(dtype=torch.float16 if self.args.fp16 else torch.float32, 
                                                            device=self.device)
                                ,feat_train],
                                dim=-1)
            
        if not full_ft or self.whole_graph:
            graph = self.graph.to(device=self.device)
            self.optimizer.zero_grad()
            feat_train = feat_train.to(dtype=torch.float32)  
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
                    feat_train[unlabel_idx, -self.n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
                    pred = self.model_gnn(graph, feat_train)

            if mode == "teacher":
                if full_ft:
                    loss = self.args.loss_weight*self.custom_train_loss(self.labels[train_pred_idx], pred[train_pred_idx]) + \
                            (1-self.args.loss_weight)*self.custom_train_loss(self.labels[train_pred_idx], out_lm[train_pred_idx])
                else:
                    loss = self.custom_train_loss(self.labels[train_pred_idx], pred[train_pred_idx])
            elif mode == "student":
                loss_gt = self.custom_train_loss(self.labels[train_pred_idx], pred[train_pred_idx])
                loss_kd = loss_kd_only(pred, teacher_output, temp)
                loss = loss_gt * (1 - alpha) + loss_kd * alpha
            else:
                raise Exception("unkown mode")

            if self.args.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            return evaluator(pred[self.train_idx], self.labels[self.train_idx]), loss.item()
        # if 0:
        #     ...
        else:
            # 强制使用子图而非全图可直接进入此分支
            # for 采样相邻节点id kernel_size为train_pred_idx， 扩充grad_padding为grad_idx
            num_batches = len(self.graph_loader)
            interval = num_batches//10
            with tqdm(total=num_batches, desc=f'train {epoch}/{self.args.n_epochs}', unit='batch', file=open(os.devnull, 'w')) as pbar:
                with self.graph_loader.enable_cpu_affinity():
                    for i, (sub_idx, train_pred_idx, blocks) in enumerate(self.graph_loader):
                        if self.args.grad_padding > 0:
                            if self.args.frozen_padding >= 0:
                                graph = dgl.node_subgraph(self.graph, sub_idx, output_device=self.device)
                                grad_idx = torch.cat([block.srcdata['_ID'] for block in blocks[-self.args.grad_padding:]], dim=0)
                                feat = feat_train[sub_idx]
                                train_idx = sub_idx[torch.isin(sub_idx, self.train_idx)]
                            else:
                                graph = blocks.to(device=self.device)
                                grad_idx = sub_idx
                                feat = feat_train[sub_idx]
                                train_idx = sub_idx[torch.isin(sub_idx, self.train_idx)]
                            
                        if len(grad_idx)>self.args.grad_size:
                            logger.info(f"grad_idx({len(grad_idx)}) sliced")
                            grad_idx = grad_idx[:self.args.grad_size]
                            
                        self.optimizer.zero_grad()
                        # feat = feat.detach()

                        if full_ft:
                            subset = Subset(self.text_data, grad_idx) 
                            dataloader = DataLoader(subset, batch_size=len(grad_idx))
                            for _, (input_ids, attention_mask) in enumerate(dataloader):
                                input_ids = input_ids.to(self.device)
                                attention_mask = attention_mask.to(self.device)
                                if self.args.fp16:
                                    with autocast():
                                        out_lm, embs = self.model_lm(input_ids, attention_mask, return_hidden=True)
                                    # embs = embs.to(torch.float16)
                                    # feat = feat.to(dtype=torch.float32)
                                else:
                                    out_lm, embs = self.model_lm(input_ids, attention_mask, return_hidden=True)
                        
                            torch.cuda.empty_cache()    
                            gc.collect()
                        feat = feat.to(dtype=torch.float32)    
                        
                        # gnn
                        if self.args.use_labels:
                            train_labels_idx = set(train_idx.tolist()) - set(train_pred_idx.tolist())
                            train_labels_idx = torch.tensor(list(train_labels_idx))
                            onehot_labels = self.cal_labels(self.n_node, self.labels, train_labels_idx)
                            if len(train_labels_idx):
                                feat[self.id_in_parent(sub_idx, train_labels_idx),
                                    -self.n_classes:] = onehot_labels[train_labels_idx]
                            if full_ft:
                                embs = torch.cat([embs, onehot_labels[grad_idx]], dim=-1)
                        
                        if full_ft:
                            if self.args.use_gpt_preds:
                                embs = torch.cat([self.gpt_preds[grad_idx].to(self.device), embs], dim=-1)
                            
                            feat = replace_rows(feat, self.id_in_parent(sub_idx, grad_idx), embs)

                        if self.args.n_label_iters > 0:
                            with torch.no_grad():
                                pred = self.model_gnn(graph, feat)
                        else:
                            # if self.args.fp16:
                            #     with autocast():
                            pred = self.model_gnn(graph, feat)

                        if self.args.n_label_iters > 0:
                            # unlabel_idx = torch.cat([train_pred_idx, self.val_idx, self.test_idx])
                            unlabel_idx = set(sub_idx.tolist()) - set(train_labels_idx.tolist())
                            unlabel_idx = torch.tensor(list(unlabel_idx))
                            for _ in range(self.args.n_label_iters):
                                pred = pred.detach()    #requires_grad为false, 梯度向前传播到此为止
                                # torch.cuda.empty_cache()
                                onehot_labels[unlabel_idx] = F.softmax(
                                    pred[self.id_in_parent(sub_idx, unlabel_idx)], dim=-1)
                                feat[self.id_in_parent(sub_idx, unlabel_idx), -self.n_classes:] \
                                    = onehot_labels[unlabel_idx]
                                pred = self.model_gnn(graph, feat)

                        if mode == "teacher":
                            if full_ft:
                                loss = self.args.loss_weight*self.custom_train_loss(
                                    self.labels[train_pred_idx],
                                    pred[self.id_in_parent(sub_idx, train_pred_idx)]
                                    ) + \
                                    (1-self.args.loss_weight)*self.custom_train_loss(
                                    self.labels[train_pred_idx],
                                    out_lm[self.id_in_parent(sub_idx, train_pred_idx)] 
                                    )
                            else:
                                loss = self.custom_train_loss(
                                    self.labels[train_pred_idx],
                                    pred[self.id_in_parent(sub_idx, train_pred_idx)]
                                    )
                        elif mode == "student":
                            loss_gt = self.custom_train_loss(pred[train_pred_idx], self.labels[train_pred_idx])
                            loss_kd = loss_kd_only(pred, teacher_output, temp)
                            loss = loss_gt * (1 - alpha) + loss_kd * alpha
                        else:
                            raise Exception("unkown mode")
                        
                        # torch.cuda.empty_cache()    
                        # gc.collect()
                        if self.args.fp16:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                        if (i-1) % interval == 0: logger.info(str(pbar))
                        pbar.update(1)

            return evaluator(pred[self.id_in_parent(sub_idx, train_idx)], self.labels[train_idx]), loss.item()


    @torch.no_grad()
    def evaluate(self, evaluator, full_ft):
        torch.cuda.empty_cache()    
        gc.collect()
        self.model_gnn.eval()
        if self.model_lm: self.model_lm.eval()

        # feat = graph.ndata["feat"]
        graph = self.graph.to(device=self.device)
            
        if full_ft and self.whole_graph:
            feat_eval = self.get_feat(self.device)
        else:
            if full_ft:
                self.feat_static = self.get_feat()      
            feat_eval = self.feat_static.to(self.device)
            
        if self.args.use_labels:
            onehot_labels = self.cal_labels(self.n_node, self.labels, self.train_idx)
            feat_eval = torch.cat([feat_eval, onehot_labels], dim=-1)
        if self.args.use_gpt_preds:
            feat_eval = torch.cat([self.gpt_preds.to(self.device), feat_eval], dim=-1)
            
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


    def run(self, n_running, rseed):
        evaluator_wrapper = lambda pred, labels: self.evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels} #onehot to cls
        )["acc"]

        # kd mode
        mode = self.args.kd_mode
        # define model and optimizer
        #e5_revgat
        self.gen_model()
        start_ep = 0
        last_is_full_ft = True
        if self.args.proceed:
            start_ep, last_is_full_ft = self.load_stat()
        
        logger.info(f"Number of all params: {self.count_params(grad_only=False)}")
        self.to_device(self.model_gnn)
        self.to_device(self.model_lm)
        
        # training loop
        total_time = 0
        best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
        final_pred = None

        accs, train_accs, val_accs, test_accs = [], [], [], []
        losses, train_losses, val_losses, test_losses = [], [], [], []
        for epoch in range(start_ep + 1, self.args.n_epochs + 1):
            
            tic = time.time()
            if mode == "student":
                teacher_output = torch.load("./{}/best_pred_run{}.pt".format(self.args.kd_dir, n_running)).cpu().cuda()
            else:
                teacher_output = None
                
            if self.args.peft_start > 0 and self.args.peft_start == epoch:
                self.switch_to('gnn_lora')    
            is_full_ft = self.args.ftmask[epoch] and not self.args.use_external_feat   
            if last_is_full_ft != is_full_ft:
                last_is_full_ft = is_full_ft
                if is_full_ft:
                    self.switch_to('full_ft')
                    if self.args.peft_start > 0 and epoch>=self.args.peft_start:
                        self.switch_to('gnn_lora')
                else:
                    self.switch_to('gnn_only')
                    if self.args.peft_start > 0 and epoch>=self.args.peft_start:
                        self.switch_to('gnn_backbone')
            self.adjust_learning_rate(epoch, is_full_ft)

            acc, loss = self.train(
                epoch,
                evaluator_wrapper,
                is_full_ft,
                mode=mode,
                teacher_output=teacher_output,
            )

            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = \
                                                                self.evaluate(evaluator_wrapper, is_full_ft)

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
                if mode == "teacher":
                    self.save_pred(final_pred, n_running, self.args.kd_dir)
                self.save_stat(epoch,f'best{rseed}')
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
            
            self.save_stat(epoch,is_full_ft,'last')

        logger.info("*" * 50)
        logger.info(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
        logger.info("*" * 50)

        if self.args.save_pred:
            os.makedirs(f"{self.args.output_dir}/cached_embs", exist_ok=True)
            torch.save(final_pred, f"{self.args.output_dir}/cached_embs/logits_seed{n_running}.pt")
            logger.warning(f"Saved logits to {self.args.output_dir}/cached_embs/logits_seed{n_running}.pt")

        return best_val_acc, final_test_acc



def main():
    set_logging()
    gbc = LM_GNN(parse_args())
    
    if not gbc.args.use_labels and gbc.args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    # load data & preprocess
    gbc.load_data()
    gbc.graph = preprocess(gbc.graph)#

    # to device
    gbc.prepare()
    logger.info(gbc.args)
    save_args(gbc.args, gbc.args.save)
    
    # run
    val_accs, test_accs = [], []

    for i in range(gbc.args.n_runs):
        rseed = gbc.args.seed + i
        seed(rseed)
        gbc.init_loader()
        val_acc, test_acc = gbc.run(i + 1, rseed)
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


if __name__ == "__main__":
    main()
