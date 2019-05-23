#!/usr/bin/env bash
set -eux

export FLAGS_sync_nccl_allreduce=1
export PATH="$PATH:/home/work/cuda-9.0/bin"
export LD_LIBRARY_PATH="/home/work/cuda-9.0/lib64"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/work/cudnn/cudnn_v7/cuda/lib64
export CUDA_VISIBLE_DEVICES=6

/mnt/lizhen21/python-paddle-13/python/bin/python -u ./train.py --use_cuda True \
                --is_distributed False\
                --use_fast_executor True \
                --weight_sharing True \
                --vocab_path ./config/baidu_vocab.txt \
                --train_filelist ./datav2/train_filelist \
                --valid_filelist ./datav2/valid_filelist \
                --validation_steps 2000 \
                --num_train_steps 1000000 \
                --warmup_steps 20000 \
                --checkpoints ./baidu_3kw_checkpoints_shuf_1 \
                --save_steps 10000 \
                --ernie_config_path ./config/ernie_config.json \
                --learning_rate 6.25e-5 \
                --weight_decay 0.01 \
                --max_seq_len 128 \
                --skip_steps 100
