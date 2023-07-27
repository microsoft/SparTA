export TASK_NAME=mnli
SP=$1
SIZE=$2
METHOD=$3
CMD_NAME=$4

KERNEL_SIZE=32x${SIZE} PIT_METHOD=${METHOD} SPARSE_RATIO=${SP} CUDA_VISIBLE_DEVICES=0 python run_${CMD_NAME}.py \
  --output_dir /tmp/kk_$TASK_NAME/ \
  --model_name_or_path TehranNLP-org/bert-base-uncased-avg-mnli \
  --task_name $TASK_NAME \
  --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 256 \
  --learning_rate 3e-5 \
  --warmup_steps 0 \
  --num_train_epochs 1 \
  --head_mask_str empty \
  --logging_steps 3000 \
  --evaluation_strategy epoch \
  --dense_pruning_method magnitude \
  --attention_pruning_method magnitude \
  --initial_threshold ${SP} \
  --final_threshold ${SP} \
  --initial_warmup 1 \
  --final_warmup 4 \
  --attention_block_rows ${SIZE} \
  --attention_block_cols 32 \
  --dense_block_rows ${SIZE} \
  --dense_block_cols 32 \
  --attention_output_with_dense False \
  --distil_teacher_name_or_path TehranNLP-org/bert-base-uncased-avg-mnli \
  --distil_temperature 1.0 \
  --regularization_final_lambda 30 \
  --regularization l1 \
  --max_steps 100