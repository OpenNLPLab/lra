#!/bin/sh
srun --partition=MMG --kill-on-bad-exit=1 -n2 --gres=gpu:2 --cpus-per-task=5 --ntasks-per-node=2 --mpi=pmi2 --job-name=s4-lra-listops --quotatype=reserved python -m train wandb=null experiment=s4-lra-listops-new trainer.gpus=2
