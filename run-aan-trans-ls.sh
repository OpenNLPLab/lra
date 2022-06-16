export HYDRA_FULL_ERROR=1
# export DATA_PATH=/mnt/lustre/share_data/hanxiaodong/lra_data

spring.submit arun --gpu \
-n1 \
--ntasks-per-node 1 \
--cpus-per-task 4 \
--partition MMG \
--quotatype spot \
--job-name=trans-ls-lra-aan \
'python -m train wandb=null experiment=trans-ls-lra-aan \
trainer.gpus=1 loader.batch_size=32 loader.num_workers=0' 
