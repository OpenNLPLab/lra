export HYDRA_FULL_ERROR=1

TASK=imdb
ARCH=trans-nystorm
# ARCH=s4
BS=20
N_LAYERS=5
D_MODEL=128
NORM=batch
seq_len=2048

# tno args 128 6 764k
EXPAND_RATIO_TNO=1
EXPAND_RATIO_GLU=1.5

# flash args 128 6 725k
FLASH_MAX_POSITION_EMBED=${seq_len}
FLASH_S=128

# flash_linear args 128 6 725k
FLASH_LINEAR_MAX_POSITION_EMBEDDINGS=${seq_len}
FLASH_LINEAR_S=128

# local global args 128 8 798k
LG_LOCAL_HEADS=4
LG_LINEAR_HEADS=4
LG_LOCAL_CHUNK_SIZE=64
## T2
USE_SOFTMAX=True
ACT_FUN="1+elu"
## T1
# USE_SOFTMAX=Fasle
# ACT_FUN="elu"

# long short args 128 6 798k
LS_ATTN_HEADS=8
LS_ATTN_WINDOW_SIZE=8
LS_ATTN_MAX_SEQ_LEN=${seq_len}

# performer args 128 6 794k
PERFORMER_HEADS=2
PERFORMER_APPROX_ATTN_DIM=32

# cosformer args 128 6 808k
COSFORMER_HEADS=8
COSFORMER_MAX_LENGTH=${seq_len}

# vanilla 128 6 795k

# linformer 
LINFORMER_MAX_SEQ_LEN=${seq_len}

# nystorm 128 5 744k

# reformer 128 8 794k
REFORMER_MAX_SEQ_LEN=${seq_len}
cards=1

# fnet 128 9 735k
FNET_MAX_POSITION_EMBEDDINGS=${seq_len}
FNET_EXPAND_RATIO=1

# synthesizer 128 8
SYNTHESIZER_MAX_SEQ_LEN=${seq_len}

python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.max_epochs=1 \
trainer.gpus=$cards dataset.l_max=${seq_len} \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true train.seed=2222 \
model.flash_max_position_embed=${FLASH_MAX_POSITION_EMBED} model.flash_s=${FLASH_S} \
model.flash_linear_max_position_embeddings=${FLASH_LINEAR_MAX_POSITION_EMBEDDINGS} model.flash_linear_s=${FLASH_LINEAR_S} \
model.lg_local_heads=${LG_LOCAL_HEADS} model.lg_linear_heads=${LG_LINEAR_HEADS} model.lg_local_chunk_size=${LG_LOCAL_CHUNK_SIZE} \
model.ls_attn_heads=${LS_ATTN_HEADS} model.ls_attn_window_size=${LS_ATTN_WINDOW_SIZE} model.ls_attn_max_seq_len=${LS_ATTN_MAX_SEQ_LEN} \
model.performer_heads=${PERFORMER_HEADS} model.performer_approx_attn_dim=${PERFORMER_APPROX_ATTN_DIM} \
model.cosformer_heads=${COSFORMER_HEADS} model.cosformer_max_length=${COSFORMER_MAX_LENGTH} \
model.use_softmax=${USE_SOFTMAX} model.act_fun=${ACT_FUN} \
# model.linformer_max_seq_len=${LINFORMER_MAX_SEQ_LEN} \
# model.reformer_max_seq_len=${REFORMER_MAX_SEQ_LEN} \
# model.fnet_max_position_embeddings=${FNET_MAX_POSITION_EMBEDDINGS} model.fnet_expand_ratio=${FNET_EXPAND_RATIO} \
# model.synthesizer_max_seq_len=${SYNTHESIZER_MAX_SEQ_LEN} \
# model.expand_ratio_tno=${EXPAND_RATIO_TNO} model.expand_ratio_glu=${EXPAND_RATIO_GLU} \



