# LEEG

## Environment
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# or pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_scatter==2.1.1 torch_sparse==0.6.17 torch_cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torchdata==0.5.1 torchmetrics==1.0.3
conda install pyg==2.3.1 -c pyg
pip install ogb
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install transformers evaluate gdown networkx colorlog accelerate accuracy
pip install peft --no-dependencies
pip install optuna # for hp search
pip install deepspeed # recommend using deepspeed if you want to finetune LM by your self
```
## Run

```bash
bash scripts/run_log.sh [--proceed]
```
