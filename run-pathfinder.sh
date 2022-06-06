export HYDRA_FULL_ERROR=1

spring.submit arun --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=s4-lra-pathfinder \
'python -m train wandb=null experiment=s4-lra-pathfinder-new \
trainer.gpus=2 loader.batch_size=50 model.dropout=0.1 model.layer.lr=0.004'
