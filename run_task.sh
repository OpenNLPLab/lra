#! /bin/bash
export CUDA_HOME='/mnt/lustre/share/cuda-10.2'
export PATH="/mnt/lustre/share/cuda-10.2/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.2/lib64/:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/mnt/lustre/share/cuda-10.2/lib64/:$LIBRARY_PATH"
export DATA_PATH=/mnt/lustre/share_data/qinzhen/lra_data

export HYDRA_FULL_ERROR=1

TASK=$1
ARCH=$2
BS=$3
N_LAYERS=$4
D_MODEL=$5
NORM=$6

# flash args
FLASH_MAX_POSITION_EMBED=$7
FLASH_S=$8
# flash_linear args
FLASH_LINEAR_MAX_POSITION_EMBEDDINGS=$9
FLASH_LINEAR_S=${10}
# local global args
LG_LOCAL_HEADS=${11}
LG_LINEAR_HEADS=${12}
LG_LOCAL_CHUNK_SIZE=${13}
# long short args
LS_ATTN_HEADS=${14}
LS_ATTN_WINDOW_SIZE=${15}
LS_ATTN_MAX_SEQ_LEN=${16}
# performer args
PERFORMER_HEADS=${17}
PERFORMER_APPROX_ATTN_DIM=${18}

echo here

cards=${19}

spring.submit arun --gpu \
-n$cards \
--ntasks-per-node $cards \
--cpus-per-task 5 \
--partition MMG \
--quotatype reserved \
--job-name=${TASK}_${ARCH} \
"python -m train wandb=null experiment=trans-${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true \
model.flash_max_position_embed=${FLASH_MAX_POSITION_EMBED} model.flash_s=${FLASH_S} \
model.flash_linear_max_position_embeddings=${FLASH_LINEAR_MAX_POSITION_EMBEDDINGS} model.flash_linear_s=${FLASH_LINEAR_S} \
model.lg_local_heads=${LG_LOCAL_HEADS} model.lg_linear_heads=${LG_LINEAR_HEADS} model.lg_local_chunk_size=${LG_LOCAL_CHUNK_SIZE} \
model.ls_attn_heads=${LS_ATTN_HEADS} model.ls_attn_window_size=${LS_ATTN_WINDOW_SIZE} model.ls_attn_max_seq_len=${LS_ATTN_MAX_SEQ_LEN} \
model.performer_heads=${PERFORMER_HEADS} model.performer_approx_attn_dim=${PERFORMER_APPROX_ATTN_DIM}"
