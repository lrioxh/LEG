#!/bin/bash
dataset=ogbn-arxiv

model_type=e5-revgat
suffix=main

output_dir=out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

# mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

if [[ "$@" == *"--proceed"* ]]; then
    python "run_optuna.py" --dataset "${dataset}" --model_type "${model_type}" --suffix "${suffix}" "$@" 2>&1 | tee -a "${output_dir}/optuna.log"
else
    # 如果没有提供 '--proceed'，则覆盖写入 optuna.log
    python "run_optuna.py" --dataset "${dataset}" --model_type "${model_type}" --suffix "${suffix}" "$@" 2>&1 | tee "${output_dir}/optuna.log"
fi