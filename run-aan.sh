export HYDRA_FULL_ERROR=1

spring.submit arun --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 4 \
--partition MMG \
--quotatype spot \
--job-name=s4-lra-aan \
'python -m train wandb=null experiment=s4-lra-aan-new \
trainer.gpus=2 loader.batch_size=20 model.layer.lr=0.002' 
