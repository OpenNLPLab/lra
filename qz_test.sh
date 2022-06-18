TASK=cifar
TASK=aan
ARCH=lg
BS=1
N_LAYERS=2
D_MODEL=64
NORM=batch
NORM=layer

# flash args
FLASH_MAX_POSITION_EMBED=0
FLASH_S=0
# flash_linear args
FLASH_LINEAR_MAX_POSITION_EMBEDDINGS=0
FLASH_LINEAR_S=0
# local global args
LG_LOCAL_HEADS=2
LG_LINEAR_HEADS=2
LG_LOCAL_CHUNK_SIZE=64
# long short args
LS_ATTN_HEADS=0
LS_ATTN_WINDOW_SIZE=0
LS_ATTN_MAX_SEQ_LEN=0
# performer args
PERFORMER_HEADS=0
PERFORMER_APPROX_ATTN_DIM=0

echo here

cards=1
USE_SOFTMAX=True
ACT_FUN=1+elu
USE_SOFTMAX=False
ACT_FUN=elu

python -m train wandb=null experiment=trans-${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true \
model.flash_max_position_embed=${FLASH_MAX_POSITION_EMBED} model.flash_s=${FLASH_S} \
model.flash_linear_max_position_embeddings=${FLASH_LINEAR_MAX_POSITION_EMBEDDINGS} model.flash_linear_s=${FLASH_LINEAR_S} \
model.lg_local_heads=${LG_LOCAL_HEADS} model.lg_linear_heads=${LG_LINEAR_HEADS} model.lg_local_chunk_size=${LG_LOCAL_CHUNK_SIZE} \
model.ls_attn_heads=${LS_ATTN_HEADS} model.ls_attn_window_size=${LS_ATTN_WINDOW_SIZE} model.ls_attn_max_seq_len=${LS_ATTN_MAX_SEQ_LEN} \
model.performer_heads=${PERFORMER_HEADS} model.performer_approx_attn_dim=${PERFORMER_APPROX_ATTN_DIM} \
model.use_softmax=${USE_SOFTMAX} model.act_fun=${ACT_FUN}