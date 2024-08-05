#!/bin/bash
dataset=ogbn-arxiv

model_type=e5-revgat
suffix=main

output_dir=out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

# mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

if [[ "$@" == *"--proceed"* ]]; then
    # 如果提供了 '--proceed'，则追加写入 log.txt
    python "main.py" "$@" 2>&1 | tee -a "${output_dir}/log.txt"
else
    # 如果没有提供 '--proceed'，则覆盖写入 log.txt
    python "main.py" "$@" 2>&1 | tee "${output_dir}/log.txt"
fi