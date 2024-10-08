from ..args import (
    DECOUPLING_GNN_LIST,
    GNN_LIST,
    LINK_PRED_DATASETS,
    LM_LIST,
    NODE_CLS_DATASETS,
    SAMPLING_GNN_LIST,
    LM_GNN_LIST
)
from .gnn_trainer import GNNDecouplingTrainer, GNNSamplingTrainer, MLPTrainer
from .link_gnn_trainer import LinkGCNTrainer, LinkGNNSamplingTrainer, LinkMLPTrainer
from .link_lm_trainer import LinkLMTrainer
from .lm_trainer import LMTrainer, TextDataset
from .lm_gnn_trainer import LM_GNN_Trainer


def get_trainer_class(model_type, dataset):
    # model_type, dataset = args.model_type, args.dataset
    if model_type in LM_GNN_LIST and dataset in NODE_CLS_DATASETS:
        return LM_GNN_Trainer
    if model_type in LM_LIST and dataset in LINK_PRED_DATASETS:
        return LinkLMTrainer
    if model_type in LM_LIST and dataset in NODE_CLS_DATASETS:
        return LMTrainer
    if model_type in GNN_LIST and dataset in NODE_CLS_DATASETS:
        if model_type in DECOUPLING_GNN_LIST:
            return GNNDecouplingTrainer
        elif model_type in SAMPLING_GNN_LIST:
            return GNNSamplingTrainer
        else:
            return MLPTrainer
    if model_type in GNN_LIST and dataset in LINK_PRED_DATASETS:
        if model_type == "GraphSAGE":
            return LinkGNNSamplingTrainer
        elif model_type == "GCN":
            return LinkGCNTrainer
        elif model_type == "MLP":
            return LinkMLPTrainer
        else:
            raise NotImplementedError(f"not implemented Trainer class for {model_type} on {dataset}")
