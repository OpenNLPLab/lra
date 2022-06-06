export HYDRA_FULL_ERROR=1

spring.submit run --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=debug \
'python -m train wandb=null experiment=trans-lra-imdb \
trainer.gpus=2 loader.batch_size=10' 