export HYDRA_FULL_ERROR=1

spring.submit arun --gpu \
-n$1 \
--ntasks-per-node $1 \
--cpus-per-task 5 \
--partition MMG \
--job-name=s4-lra-listops \
'python -m train wandb=null experiment=s4-lra-listops-new \
trainer.gpus=2' 
