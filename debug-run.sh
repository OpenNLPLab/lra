export HYDRA_FULL_ERROR=1

TASK=imdb
ARCH=tno
BS=20
N_LAYERS=6
D_MODEL=128
NORM=batch
EXPAND_RATIO_TNO=1
EXPAND_RATIO_GLU=1.5

cards=1

seq_len=5000

# spring.submit run --gpu \
# -n$cards \
# --ntasks-per-node $cards \
# --cpus-per-task 5 \
# --partition MMG \
# --quotatype spot \
# --job-name=${TASK}_${ARCH} \
python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.max_epochs=1 \
trainer.gpus=$cards dataset.l_max=${seq_len} \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true train.seed=2222 \
model.expand_ratio_tno=${EXPAND_RATIO_TNO} model.expand_ratio_glu=${EXPAND_RATIO_GLU}

# python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
# trainer.gpus=$cards \
# loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true train.seed=2222

# python -m train wandb=null experiment=${ARCH}-lra-${TASK} loader.batch_size=${BS} dataset.l_max=${seq_len} trainer.max_epochs=1
