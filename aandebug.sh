export HYDRA_FULL_ERROR=1
export JOBLIB_TEMP_FOLDER=/mnt/cache/hanxiaodong/lra/data/tmp
# export DATA_PATH=/mnt/lustre/share_data/hanxiaodong/lra_data

spring.submit run --gpu \
-n1 \
--ntasks-per-node 1 \
--cpus-per-task 4 \
--partition MMG \
--quotatype spot \
--job-name=trans-lg-lra-aan \
"python -m train wandb=null experiment=trans-lg-lra-aan \
trainer.gpus=1 loader.batch_size=20 loader.num_workers=0"
