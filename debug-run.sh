export HYDRA_FULL_ERROR=1

TASK=cifar
ARCH=tno
BS=16
N_LAYERS=2
D_MODEL=64
NORM=layer
EXPAND_RATIO_TNO=1
EXPAND_RATIO_GLU=1

cards=1

# spring.submit arun --gpu \
# -n$cards \
# --ntasks-per-node $cards \
# --cpus-per-task 5 \
# --partition MMG \
# --quotatype spot \
# --job-name=${TASK}_${ARCH} \
python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true train.seed=2222 \
model.expand_ratio_tno=${EXPAND_RATIO_TNO} model.expand_ratio_glu=${EXPAND_RATIO_GLU}

# python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
# trainer.gpus=$cards \
# loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true train.seed=2222