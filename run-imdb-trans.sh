export HYDRA_FULL_ERROR=1

# spring.submit arun --gpu \
# -n2 \
# --ntasks-per-node 2 \
# --cpus-per-task 5 \
# --partition MMG \
# --quotatype auto \
# --job-name=trans-lra-imdb \
python -m train wandb=null experiment=trans-lra-imdb \
trainer.gpus=1 loader.batch_size=5
