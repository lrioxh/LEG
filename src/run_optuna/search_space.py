import logging
import warnings

from optuna.exceptions import ExperimentalWarning

from .HP_search import Dist_HP_search, Single_HP_search, Sample_HP_search

from main import LM_GNN, seed

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


class Decoupling_GNN_HP_search(Dist_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-5, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_label_smoothing = trial.suggest_float("gnn_label_smoothing", 0.1, 0.7)
        args.gnn_warmup_ratio = trial.suggest_float("gnn_warmup_ratio", 0.1, 0.5)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 4, 8)
        return args


class LM_HP_search(Dist_HP_search):
    def setup_search_space(self, args, trial):
        args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.7)
        args.accum_interval = trial.suggest_categorical("accum_interval", [1, 5, 10])
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.5)
        args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)
        return args


class PEFT_LM_HP_search(Dist_HP_search):
    def setup_search_space(self, args, trial):
        args.lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.7)
        args.peft_r = trial.suggest_categorical("peft_r", [1, 2, 4, 8])
        args.peft_lora_alpha = trial.suggest_categorical("peft_lora_alpha", [4, 8, 16, 32])
        args.peft_lora_dropout = trial.suggest_float("peft_lora_dropout", 0.1, 0.8)
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.8)
        return args


class Sampling_GNN_HP_search(Single_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-4, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_label_smoothing = trial.suggest_float("gnn_label_smoothing", 0.1, 0.7)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 2, 3)
        return args


class Link_GNN_HP_search(Single_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-4, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 2, 3)
        return args

class LM_GNN_HP_search(Single_HP_search):
    def setup_search_space(self, args, trial):
        # args.gm_lr = trial.suggest_float("gm_lr", 3e-3, 0.1, log=True)
        args.gm_lr = trial.suggest_float("gm_lr", 1e-4, 1e-3, log=True)
        args.lm_lr = trial.suggest_float("lm_lr", 5e-5, 1e-3, log=True)
        # args.wd = trial.suggest_float("wd", 1e-6, 1e-4, log=True)
        # args.wd = trial.suggest_categorical("wd", [0, 5e-6])
        # args.wu_lm = trial.suggest_categorical("wu_lm", [0, 1])
        # args.eps = trial.suggest_categorical("eps", [None, 1e-3])
        # args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        # args.ep_gm = trial.suggest_categorical("ep_gm", [18,14,10,6])
        # args.warmup = trial.suggest_categorical("warmup", [10, 20, 30])
        # args.kernel_size = trial.suggest_categorical("kernel_size", [2, 4, 8])
        args.secsam_method = trial.suggest_categorical("secsam_method", ["nearby","morehop"])
        # args.frozen_padding = trial.suggest_categorical("frozen_padding", [0, 1, 3])
        # args.peft_r_gm = trial.suggest_categorical("peft_r_gm", [4,8])
        # args.peft_r_lm = trial.suggest_categorical("peft_r_lm", [4,8])
        # args.peft_start = trial.suggest_categorical("peft_start", [12,50])
        # args.peft_start = trial.suggest_categorical("use_default_config", [True, False])
        return args
    
    def train(self, args, trial=None):
        gbc = LM_GNN(args, trial=trial)
        # load data & preprocess
        gbc.load_data()
        gbc.preprocess()#

        # to device
        gbc.prepare()

        seed(args.seed)
        gbc.init_loader()
        val_acc, test_acc = gbc.run(1, args.seed, args.prune_tolerate)

        return val_acc
    
class LM_GNN_HP_sample(Sample_HP_search):
    def setup_search_space(self, args, trial = None):
        search_space = {
            'group_1': [
                {'ep_gm': 18, 'warmup': 10},
                {'ep_gm': 14, 'warmup': 8},
                {'ep_gm': 10, 'warmup': 6},
            ],
            'group_2': [
                {'gm_lr': 6e-4, 'lm_lr': 5e-4},
                {'gm_lr': 3e-4, 'lm_lr': 2e-4},
            ]
        }
        return search_space
    
    def train(self, args, trial=None):
        gbc = LM_GNN(args, trial=trial)
        # load data & preprocess
        gbc.load_data()
        gbc.preprocess()#

        # to device
        gbc.prepare()

        seed(args.seed)
        gbc.init_loader()
        val_acc, test_acc = gbc.run(1, args.seed, args.prune_tolerate)

        return val_acc