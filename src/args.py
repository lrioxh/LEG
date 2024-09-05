import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

LM_LIST = [
    "all-roberta-large-v1",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "e5-large",
    "deberta-base",
    "deberta-v2-xxlarge",
    "deberta-v3-base",
    "sentence-t5-large",
    "roberta-large",
    "instructor-xl",
    "e5-large-v2",
]
GNN_LIST = ["GAMLP", "SAGN", "SIGN", "SGC", "GraphSAGE", "GCN", "MLP"]
LM_GNN_LIST = ["e5-revgat"]

SAMPLING_GNN_LIST = ["GraphSAGE", "GCN"]
DECOUPLING_GNN_LIST = ["GAMLP", "SAGN", "SIGN", "SGC"]

LINK_PRED_DATASETS = ["ogbl-citation2"]
NODE_CLS_DATASETS = ["ogbn-arxiv", "ogbn-products", "ogbn-arxiv-tape"]

def parse_args():
    parser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--proceed", action="store_true", default=False, help="Continue to train on presaved ckpt")
    parser.add_argument("--suffix", type=str, default="main")
    parser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--n_runs", type=int, default=1, help="running times")
    parser.add_argument("--ep_blocks", type=int, default=1, help="number of epoch blocks")     
    parser.add_argument("--ep_gm", type=int, default=18, help="number of epochs for GM only in one block")   
    parser.add_argument("--ep_full", type=int, default=2, help="number of epochs for full tuning in one block")   
    parser.add_argument("--eval_epoch", type=int, default=1) 
    parser.add_argument("--gm_lr", type=float, default=3e-4, help="learning rate for GM")
    parser.add_argument("--lm_lr", type=float, default=2e-4, help="learning rate for LM")
    parser.add_argument("--wd", type=float, default=5e-6, help="weight decay")    
    parser.add_argument("--warmup", type=int, default=10, help="epochs for warmup")    
    parser.add_argument("--wu_lm", type=int, default=0, help="epochs for warmup for LM only")  
    parser.add_argument("--loss_reduction", type=str, default='mean', help="Specifies the reduction to apply to the loss output")  
    parser.add_argument("--loss_weight", type=float, default=0.5, help="weight of full loss in the conbined loss")   
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "constant"],
    )
    parser.add_argument("--label_smoothing_factor", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")   
    
    # sampling
    parser.add_argument("--kernel_size", type=int, default=8, help="for trainable node kernel")
    parser.add_argument("--grad_padding", type=int, default=1, help="padding hop for grad scope, -1 means whole graph")
    parser.add_argument("--grad_k", type=int, default=1, help="Number of nodes sampled per hop, -1 means all neighbours")
    parser.add_argument("--grad_size", type=int, default=16, help="Max Grad Size")
    parser.add_argument("--secsam_method", type=str, default="nearby",choices=["nearby","morehop"])
    parser.add_argument("--frozen_padding", type=int, default=3, help="padding size for frozen scope, -1 means whole graph")

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
    
    # LM    
    parser.add_argument("--batch_size_infer", type=int, default=256, help="for LM static embedding")
    parser.add_argument("--batch_size_train", type=int, default=16, help="for LM warming up")
    # parser.add_argument("--batch_size_eval", type=int, default=200)
    parser.add_argument("--accum_interval", type=int, default=5)    #for LM
    parser.add_argument("--use_default_config", action="store_true", default=False)
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="consistent with the one in bert",
    )
    parser.add_argument("--header_dropout_prob", type=float, default=0.6)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adapter_hidden_size", type=int, default=768)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--avg_alpha", type=float, default=0.5)
    parser.add_argument("--eval_delay", type=int, default=0)
    parser.add_argument("--use_cache", action="store_true", default=False)
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--eval_patience", type=int, default=50000)
    parser.add_argument("--warmup_ratio", type=float, default=0.15)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "save_bert_x"]
    )
    parser.add_argument("--deepspeed", type=str, default="ds_config.json")
    parser.add_argument("--report_to", type=str, default="none")
    
    # parameters for data and model storage
    parser.add_argument("--model_type", type=str, default="e5-revgat")
    parser.add_argument("--data_folder", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--task_type", type=str, default="node_cls")
    parser.add_argument("--ckpt_dir", type=str, default='', help="path to load gnn ckpt")
    parser.add_argument("--output_dir", type=str, default=f"out")        
    parser.add_argument(
        "--use_labels", action="store_true", default=False, help="Use labels in the training set as input features."
    )
    parser.add_argument("--use_gpt_preds", action="store_true", default=False)
    parser.add_argument("--n_gpt_embs", type=int, default=64)
    parser.add_argument("--use_external_feat", action="store_true", default=False, help="use external static features")
    parser.add_argument("--feat_dir", type=str,default="out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt", help="path for external static features")
    
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
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="hidden size of bert-like model"
    )
     
    # peft & lora hyperparams
    parser.add_argument("--peft_start", type=int, default=-1, help='epoch that start to train GM with PEFT')
    parser.add_argument("--use_peft", action="store_true", default=False)
    parser.add_argument("--peft_r_lm", type=int, default=8)
    parser.add_argument("--peft_r_gm", type=int, default=8)
    parser.add_argument("--peft_lora_alpha", type=float, default=8)
    parser.add_argument("--peft_lora_dropout", type=float, default=0.3)
    
    # optuna
    parser.add_argument("--expected_valid_acc", type=float, default=0)
    parser.add_argument("--prune_tolerate", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=18)
    parser.add_argument("--load_study", action="store_true", default=False)
    
    args = parser.parse_args()
    args = _set_lm_and_gnn_type(args)
    args = _set_dataset_specific_args(args)
    args = _set_pretrained_repo(args)
    args.save = f"{args.output_dir}/{args.dataset}/{args.model_type}/{args.suffix}"
    os.makedirs(f"{args.save}/ckpt",exist_ok=True)
    # 可以直接从这里控制：|0: ep从1开始|0 for gnn & 1 for lm+gnn|
    args.ftmask = [0, 1]+([1 for _ in range(args.ep_full)]+[0 for _ in range(args.ep_gm)])*args.ep_blocks
    if args.peft_start>0:
        args.ftmask = args.ftmask +[1]*2 + [0]*8
    # args.ftmask = [0]+([0 for _ in range(args.ep_gm)]+[1 for _ in range(args.ep_full)])*args.ep_blocks+[0]*6+[1]*2
    args.n_epochs = len(args.ftmask)-1
    args.no_attn_dst = True
    args.use_peft = True
    args.fp16 = True
    args.use_labels = True
    # args.use_gpt_preds = True
    args.debug = 80000
    # args.proceed = True
    # args.use_external_feat = True
    # args.train_idx_cluster = True
    args.use_default_config = True
    args.deepspeed = None
    args.disable_tqdm = True
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

def _set_lm_and_gnn_type(args):
    if "e5" in args.model_type:
        args.lm_type = "e5-large"
    elif "de" in args.model_type:
        args.lm_type = "deberta-v3-base"
    
    if "revgat" in args.model_type:
        args.gnn_type = "RevGAT"
    elif "sage" in args.model_type:
        args.gnn_type = "GraphSAGE"
    return args

def _set_pretrained_repo(args):
    dict = {
        "all-roberta-large-v1": "sentence-transformers/all-roberta-large-v1",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-large": "intfloat/e5-large",    #512
        "deberta-base": "microsoft/deberta-base",   #3000+
        "deberta-v2-xxlarge": "microsoft/deberta-v2-xxlarge",   #2001
        "deberta-v3-base": "microsoft/deberta-v3-base",     #2306
        "sentence-t5-large": "sentence-transformers/sentence-t5-large",
        "roberta-large": "roberta-large",
        "instructor-xl": "hkunlp/instructor-xl",
        "e5-large-v2": "intfloat/e5-large-v2",
    }

    if args.model_type in dict.keys():
        args.pretrained_repo = dict[args.model_type]
    elif args.lm_type in dict.keys():
        args.pretrained_repo = dict[args.lm_type]
    else:
        assert args.lm_type in dict.keys() or args.model_type in dict.keys()
        # assert args.pretrained_repo in dict[args.lm_type]
    return args



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

    hidden_size_dict = {
        "all-roberta-large-v1": 1024,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384,
        "e5-large": 1024,
        "e5-large-v2": 1024,
        "deberta-v2-xxlarge": 1536,
        "deberta-v3-base": 768,
        "sentence-t5-large": 768,
        "roberta-large": 1024,
        "instructor-xl": 1024,
    }

    if args.lm_type in hidden_size_dict.keys():
        args.hidden_size = hidden_size_dict[args.lm_type]
    # elif args.use_bert_x and args.lm_type in hidden_size_dict.keys():
    #     args.num_feats = args.hidden_size = hidden_size_dict[args.lm_type]
    # elif args.use_giant_x:
    #     args.num_feats = args.hidden_size = 768
    # elif args.use_gpt_preds:
    #     args.num_feats = 5

    return args

