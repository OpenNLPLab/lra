export HYDRA_FULL_ERROR=1

TASK=$1
ARCH=$2
BS=$3
N_LAYERS=$4
D_MODEL=$5
TNO_DPB_DIM=$6
NORM=$7
EXPAND_RATIO_TNO=$8
EXPAND_RATIO_GLU=$9
TNO_USE_DECAY=${10}
TNO_GAMMA=${11}
cards=${12}
spring.submit arun --gpu \
-n$cards \
--ntasks-per-node $cards \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=${TASK}_${ARCH}_${EXPAND_RATIO_TNO}_${EXPAND_RATIO_GLU}_${NORM} \
"python -m train wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} model.tno_dpb_dim=${TNO_DPB_DIM} \
model.norm=${NORM} model.prenorm=true train.seed=2222 \
model.expand_ratio_tno=${EXPAND_RATIO_TNO} model.expand_ratio_glu=${EXPAND_RATIO_GLU} \
model.tno_use_decay=${TNO_USE_DECAY} model.tno_gamma=${TNO_GAMMA}"