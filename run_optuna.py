import gc
import logging
import os
import sys
import warnings

from optuna.exceptions import ExperimentalWarning

from src.args import parse_args, save_args
from src.run_optuna.search_space import (
    Decoupling_GNN_HP_search,
    LM_HP_search,
    PEFT_LM_HP_search,
    Sampling_GNN_HP_search,
    LM_GNN_HP_search,
    LM_GNN_HP_sample
)
from src.utils import set_logging

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


def get_search_instance(model_type, use_peft=False, sample_hp = False):
    if sample_hp: 
        return LM_GNN_HP_sample
    if model_type in [
        "all-roberta-large-v1",
        "all-mpnet-base-v2",
        "all-MiniLM-L6-v2",
        "e5-large",
        "deberta-v2-xxlarge",
    ]:
        return PEFT_LM_HP_search if use_peft else LM_HP_search
    elif model_type in ["GAMLP", "SAGN", "SGC"]:
        return Decoupling_GNN_HP_search
    elif model_type in ["GraphSAGE", "GCN"]:
        return Sampling_GNN_HP_search
    elif lambda s, prefixes, suffixes: s.count('-')==1 and s.split('-')[0] in prefixes and s.split('-')[1] in suffixes\
                (model_type, ["e5", "de"], ["revgat", "sage"]):
        return LM_GNN_HP_search
    else:
        raise NotImplementedError("not implemented HP search class")


def main():
    set_logging()
    args = parse_args()
    hp_search = get_search_instance(args.model_type, args.use_peft, args.sample_hp)(args)
    if args.load_study:
        hp_search.load_study()
    else:
        logger.critical(
            f"Start HP search, optuna the {args.model_type} model on {args.dataset} dataset for {args.n_trials} trials"
        )
        hp_search.run(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
