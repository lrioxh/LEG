#!/bin/bash
dataset=ogbn-products

model_type=e5-sage
suffix=main
# gm_lr=0.008
output_dir=out/${dataset}/${model_type}/${suffix}
mkdir -p ${output_dir}
if [[ "$@" == *"--proceed"* ]]; then
    python "main.py" \
    --dataset "${dataset}" \
    --model_type "${model_type}" \
    --suffix "${suffix}" \
    "$@" 2>&1 | tee -a "${output_dir}/run.log"
else
    python "main.py" \
    --dataset "${dataset}" \
    --model_type "${model_type}" \
    --suffix "${suffix}" \
    "$@" 2>&1 | tee "${output_dir}/run.log"
fi