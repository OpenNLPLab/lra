2export HYDRA_FULL_ERROR=1
TASK=cifar
ARCH=flash
BS=10
N_LAYERS=6
D_MODEL=512
NORM=batch

# flash args
FLASH_MAX_POSITION_EMBED=1024
FLASH_S=128
# flash_linear args
FLASH_LINEAR_MAX_POSITION_EMBEDDINGS=0
FLASH_LINEAR_S=0
# local global args
LG_LOCAL_HEADS=0
LG_LINEAR_HEADS=0
LG_LOCAL_CHUNK_SIZE=0
# long short args
LS_ATTN_HEADS=0
LS_ATTN_WINDOW_SIZE=0
LS_ATTN_MAX_SEQ_LEN=0
# performer args
PERFORMER_HEADS=0
PERFORMER_APPROX_ATTN_DIM=0

spring.submit run --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=${TASK}_${ARCH} \
"python -m train wandb=null experiment=trans-${ARCH}-lra-${TASK} \
trainer.gpus=2 \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true \
model.flash_max_position_embed=${FLASH_MAX_POSITION_EMBED} model.flash_s=${FLASH_S} \
model.flash_linear_max_position_embeddings=${FLASH_LINEAR_MAX_POSITION_EMBEDDINGS} model.flash_linear_s=${FLASH_LINEAR_S} \
model.lg_local_heads=${LG_LOCAL_HEADS} model.lg_linear_heads=${LG_LINEAR_HEADS} model.lg_local_chunk_size=${LG_LOCAL_CHUNK_SIZE} \
model.ls_attn_heads=${LS_ATTN_HEADS} model.ls_attn_window_size=${LS_ATTN_WINDOW_SIZE} model.ls_attn_max_seq_len=${LS_ATTN_MAX_SEQ_LEN} \
model.performer_heads=${PERFORMER_HEADS} model.performer_approx_attn_dim=${PERFORMER_APPROX_ATTN_DIM}"