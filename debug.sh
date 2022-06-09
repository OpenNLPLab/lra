export HYDRA_FULL_ERROR=1

spring.submit run --gpu \
-n1 \
--ntasks-per-node 1 \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=debug \
'python -m train wandb=null experiment=trans-linear-lra-listops \
trainer.gpus=1 loader.batch_size=10 ' 