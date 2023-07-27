MODEL_NAME='mf-lmd6remi-f1'  # v2.3.1
# bos token is regarded as a fake bar, add beat embedding
# (16,141,312  16,141,312)

DATA_DIR=data-bin/lmd6remi      # Data dir

# 4 GPUs
BATCH_SIZE=1
UPDATE_FREQ=1

PEAK_LR=5e-4            # Peak learning rate, adjust as needed
WARMUP_UPDATES=16000     # Warmup the learning rate over this many updates


OMP_NUM_THREADS=$(cat /proc/cpuinfo| grep "processor"| wc -l)

ulimit -n 4096

mkdir -p log

#python -m torch.distributed.launch \
#  --nproc_per_node=8 \
#  --nnodes=2 \
#  --node_rank=0 \
#  --master_addr="192.168.0.88" \
#  --master_port=12345 \
METHOD=PyTorch TRUNCATE_TRAIN=$1 CUDA_VISIBLE_DEVICES=1 $(which fairseq-train) \
  $DATA_DIR \
  --user-dir museformer \
  --task museformer_language_modeling \
  --arch museformer_lm_v2s1 \
  --sum2sum-self full \
  --num-layers 3 \
  --num-attention-heads '(8,)' \
  --attention-embed-dim 512 \
  --ffn-embed-dim 2048 \
  --block-size 64 \
  --take-bos-as-bar True \
  --num-summary-tokens-each-chunk 1 \
  --use-token-in-chunk-abs-pos False \
  --use-token-abs-pos False \
  --use-token-rel-pos False \
  --use-bar-abs-pos True \
  --max-bar-abs-pos 512 \
  --bar-abs-pos-embed-dim 256 \
  --use-bar-rel-pos False \
  --use-beat-abs-pos True \
  --max-beat-abs-pos 64 \
  --beat-abs-pos-embed-dim 128 \
  --tokens-per-sample 100000 \
  --truncate-train $1 \
  --truncate-valid 10240 \
  --batch-size $BATCH_SIZE \
  --update-freq $UPDATE_FREQ \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --adam-eps 1e-9 \
  --weight-decay 0.01 \
  --lr $PEAK_LR \
  --lr-scheduler inverse_sqrt \
  --warmup-updates $WARMUP_UPDATES \
  --max-update 1000000 \
  --validate-interval 1000000000 \
  --save-interval 1000000000 \
  --save-interval-updates 5000 \
  --log-format simple \
  --log-interval 10 \
  --num-workers "$OMP_NUM_THREADS" \
  --save-dir checkpoints/$MODEL_NAME \
  --attention-impl mask \
  | tee log/${MODEL_NAME}.log
