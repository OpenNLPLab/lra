export HYDRA_FULL_ERROR=1

spring.submit arun --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 4 \
--partition MMG \
--quotatype spot \
--job-name=trans-flash-lra-cifar \
'python -m train wandb=null experiment=trans-flash-lra-cifar \
trainer.gpus=2 loader.batch_size=10 ' 