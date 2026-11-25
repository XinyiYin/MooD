#!/bin/bash

set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "============================================"
echo "Detected ${NUM_GPUS} GPU(s)"
echo "============================================"

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: No GPU detected"
    exit 1
fi

GRAD_ACCUM=8
if [ "$NUM_GPUS" -eq 2 ]; then
    GRAD_ACCUM=4
    echo "2-GPU mode: grad_accum=4, effective_batch=8"
elif [ "$NUM_GPUS" -eq 4 ]; then
    GRAD_ACCUM=2
    echo "4-GPU mode: grad_accum=2, effective_batch=8"
elif [ "$NUM_GPUS" -ge 6 ]; then
    GRAD_ACCUM=1
    echo "${NUM_GPUS}-GPU mode: grad_accum=1, effective_batch=${NUM_GPUS}"
else
    echo "1-GPU mode: grad_accum=8, effective_batch=8"
fi

echo "============================================"
echo ""



# TODO: supplement the file path

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    --multi_gpu \
    train.py \
    --pretrained_model  \
    --train_data_dir  \
    --output_dir  \
    --attr_list_path  \
    --attr_va_stats_path  \
    --learning_rate 3e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 300 \
    --max_train_steps 78000 \
    --train_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --mixed_precision bf16 \
    --max_grad_norm 1.0 \
    --recon_weight 1.0 \
    --attr_weight 1.0 \
    --va_weight 0.1 \
    --num_vectors 4 \
    --num_mapper_layers 3 \
    --mapper_dropout 0.2 \
    --placeholder_token "<emo>" \
    --initializer_token "emotion" \
    --latent_cache_dirs "   " \
    --emo_cache_dir "   " \
    --resolution 1024 \
    --dataloader_num_workers 8 \
    --checkpointing_steps 3000 \
    --val_every_n_epochs 1 \
    --patience 5 \
    --seed 42