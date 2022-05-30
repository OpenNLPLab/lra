#!/bin/sh
srun --partition=MMG --kill-on-bad-exit=1 -n4 --gres=gpu:4 --cpus-per-task=5 --ntasks-per-node=4 --mpi=pmi2 --job-name=s4-lra-imdb --quotatype=reserved python -m train wandb=null experiment=s4-lra-cifar-new trainer.gpus=4
