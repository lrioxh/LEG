#!/bin/bash
dataset=ogbn-arxiv

model_type=e5-revgat
suffix=main

output_dir=out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

# mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

if [[ "$@" == *"--proceed"* ]]; then
    python "main.py" --dataset "${dataset}" --model_type "${model_type}" --suffix "${suffix}" "$@" 2>&1 | tee -a "${output_dir}/run.log"
else
    # 如果没有提供 '--proceed'，则覆盖写入 run.log
    python "main.py" --dataset "${dataset}" --model_type "${model_type}" --suffix "${suffix}" "$@" 2>&1 | tee "${output_dir}/run.log"
fi