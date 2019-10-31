set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0

python -u run_classifier.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val false \
                   --do_test true \
                   --batch_size 128 \
                   --init_pretraining_params ernie_checkpoint/params/ \
                   --train_set ChnSentiCorp_data/train.tsv \
                   --dev_set ChnSentiCorp_data/dev.tsv \
                   --test_set ChnSentiCorp_data/test.tsv \
                   --vocab_path config/vocab.txt \
                   --checkpoints model_checkpoint \
                   --save_steps 2000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --validation_steps 80 \
                   --epoch 5 \
                   --batch_merge_repeat 4 \
                   --max_seq_len 128 \
                   --ernie_config_path config/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1
