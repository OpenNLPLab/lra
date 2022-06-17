export HYDRA_FULL_ERROR=1

spring.submit arun --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=trans-flash-lra-listops \
'python -m train wandb=null experiment=trans-flash-lra-listops \
trainer.gpus=2 loader.batch_size=10 ' 
