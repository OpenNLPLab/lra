export HYDRA_FULL_ERROR=1
export DATA_PATH=/mnt/lustre/share_data/hanxiaodong/lra_data

spring.submit arun --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 4 \
--partition MMG \
--quotatype spot \
--job-name=trans-linear-lra-aan \
'python -m train wandb=null experiment=trans-linear-lra-aan \
trainer.gpus=2 loader.batch_size=20 loader.num_workers=0' 
