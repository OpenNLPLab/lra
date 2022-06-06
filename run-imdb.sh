export HYDRA_FULL_ERROR=1

spring.submit arun --gpu \
-n2 \
--ntasks-per-node 2 \
--cpus-per-task 5 \
--partition MMG \
--quotatype spot \
--job-name=s4-lra-imdb \
'python -m train wandb=null experiment=s4-lra-imdb-new \
trainer.gpus=2 loader.batch_size=25 model.n_layers=4 model.layer.lr=0.001' 
