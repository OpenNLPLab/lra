ARCH=$1
TASK=$2

cards=$3

spring.submit arun --gpu \
-n$cards \
--ntasks-per-node $cards \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=${ARCH}_${TASK} \
"python -m train wandb=null experiment=${TASK}-lra-${ARCH} trainer.gpus=$cards"