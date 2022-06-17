#! /bin/bash
export CUDA_HOME='/mnt/lustre/share/cuda-10.2'
export PATH="/mnt/lustre/share/cuda-10.2/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.2/lib64/:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/mnt/lustre/share/cuda-10.2/lib64/:$LIBRARY_PATH"
export DATA_PATH=/mnt/lustre/share_data/qinzhen/lra_data

echo $CUDA_HOME

TASK=aan
ARCH=lg
num_gpu=2

spring.submit arun --gpu \
-n$num_gpu \
--ntasks-per-node $num_gpu \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=$TASK_$ARCH \
"python -m train wandb=null experiment=trans-lg-lra-aan \
 trainer.gpus=$num_gpu loader.batch_size=32"

#  loader.num_workers=0