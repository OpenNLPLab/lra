export HYDRA_FULL_ERROR=1

TASK=cifar
ARCH=tno-v2
BS=16
N_LAYERS=12
D_MODEL=192
NORM=layer

cards=2

spring.submit arun --gpu \
-n$cards \
--ntasks-per-node $cards \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=${TASK}_${ARCH} \
"python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true train.seed=2222"

# python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
# trainer.gpus=$cards \
# loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.norm=${NORM} model.prenorm=true train.seed=2222