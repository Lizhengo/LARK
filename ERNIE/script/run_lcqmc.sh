set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0

python -u run_classifier.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 64 \
                   --init_pretraining_params ./ernie_checkpoint/params/ \
                   --train_set ${TRAIN_DATA_PATH} \
                   --dev_set ${DEV_DATA_PATH} \
                   --test_set ${TEST_DATA_PATH} \
                   --vocab_path config/vocab.txt \
                   --checkpoints ./model_checkpoint \
                   --save_steps 2000 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.0 \
                   --validation_steps 2000 \
                   --epoch 10 \
                   --max_seq_len 128 \
                   --ernie_config_path config/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1
