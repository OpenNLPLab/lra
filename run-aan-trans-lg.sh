export HYDRA_FULL_ERROR=1
export export JOBLIB_TEMP_FOLDER=/mnt/DATA/hxd/lra/tmpdir
# export DATA_PATH=/mnt/lustre/share_data/hanxiaodong/lra_data

# spring.submit arun --gpu \
# -n2 \
# --ntasks-per-node 2 \
# --cpus-per-task 4 \
# --partition MMG \
# --quotatype spot \
# --job-name=trans-lg-lra-aan \
python -m train wandb=null experiment=trans-lg-lra-aan \
trainer.gpus=1 loader.batch_size=20 loader.num_workers=0
