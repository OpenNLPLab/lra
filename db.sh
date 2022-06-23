export HYDRA_FULL_ERROR=1

TASK='imdb'
ARCH='reformer'
ARCH='cosformer'
BS=20
N_LAYERS=6
D_MODEL=128
NORM='batch'

# flash args
FLASH_MAX_POSITION_EMBED=0
FLASH_S=0
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

USE_SOFTMAX=0
ACT_FUN=0

# cosformer args
COSFORMER_HEADS=2
COSFORMER_MAX_LENGTH=2048

# linformer args
LINFORMER_MAX_SEQ_LEN=5000

# reformer args
REFORMER_MAX_SEQ_LEN=2048

# nystorm args
NYSTORM_MAX_SEQ_LEN=2048

cards=1

# spring.submit run --gpu \
# -n$cards \
# --ntasks-per-node $cards \
# --cpus-per-task 5 \
# --partition MMG \
# --quotatype spot \
# --job-name=${TASK}_${ARCH} \
# "python -m train wandb=null experiment=trans-${ARCH}-lra-${TASK} \
# trainer.gpus=$cards \
# loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true \
# model.flash_max_position_embed=${FLASH_MAX_POSITION_EMBED} model.flash_s=${FLASH_S} \
# model.flash_linear_max_position_embeddings=${FLASH_LINEAR_MAX_POSITION_EMBEDDINGS} model.flash_linear_s=${FLASH_LINEAR_S} \
# model.lg_local_heads=${LG_LOCAL_HEADS} model.lg_linear_heads=${LG_LINEAR_HEADS} model.lg_local_chunk_size=${LG_LOCAL_CHUNK_SIZE} \
# model.ls_attn_heads=${LS_ATTN_HEADS} model.ls_attn_window_size=${LS_ATTN_WINDOW_SIZE} model.ls_attn_max_seq_len=${LS_ATTN_MAX_SEQ_LEN} \
# model.performer_heads=${PERFORMER_HEADS} model.performer_approx_attn_dim=${PERFORMER_APPROX_ATTN_DIM} \
# model.cosformer_heads=${COSFORMER_HEADS} model.cosformer_max_length=${COSFORMER_MAX_LENGTH} \
# model.use_softmax=${USE_SOFTMAX} model.act_fun=${ACT_FUN} "

python -m train wandb=null experiment=trans-${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true dataset.l_max=${REFORMER_MAX_SEQ_LEN} \
model.flash_max_position_embed=${FLASH_MAX_POSITION_EMBED} model.flash_s=${FLASH_S} \
model.flash_linear_max_position_embeddings=${FLASH_LINEAR_MAX_POSITION_EMBEDDINGS} model.flash_linear_s=${FLASH_LINEAR_S} \
model.lg_local_heads=${LG_LOCAL_HEADS} model.lg_linear_heads=${LG_LINEAR_HEADS} model.lg_local_chunk_size=${LG_LOCAL_CHUNK_SIZE} \
model.ls_attn_heads=${LS_ATTN_HEADS} model.ls_attn_window_size=${LS_ATTN_WINDOW_SIZE} model.ls_attn_max_seq_len=${LS_ATTN_MAX_SEQ_LEN} \
model.performer_heads=${PERFORMER_HEADS} model.performer_approx_attn_dim=${PERFORMER_APPROX_ATTN_DIM} \
model.cosformer_heads=${COSFORMER_HEADS} model.cosformer_max_length=${COSFORMER_MAX_LENGTH} \
model.linformer_max_seq_len=${LINFORMER_MAX_SEQ_LEN} \
# model.reformer_max_seq_len=${REFORMER_MAX_SEQ_LEN} \
model.use_softmax=${USE_SOFTMAX} model.act_fun=${ACT_FUN} trainer.max_epochs=5|tee debug/${ARCH}_${REFORMER_MAX_SEQ_LEN}.log

# model.nystorm_max_seq_len=${NYSTORM_MAX_SEQ_LEN} \

