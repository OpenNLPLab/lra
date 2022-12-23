import os
import sys
import time
from turtle import pd
import signal
import subprocess

EXPAND_RATIO_gtu=1
EXPAND_RATIO_GLU=1

cards=1
tasks = ["imdb","listops","pathfinder","pathfinderx","aan"]
tasks = ['pathfinderx']
archs = ["gtu"]
archs = ['gtu2d']
bs = {'cifar':25, 'imdb':16, 'listops':16, 'pathfinder':64, 'pathfinderx':16, 'aan':16}
# change
n_layers = 2
d_model = 64
gtu_dpb_dim = min(d_model//4, 64)
# change
norm = 'batch'
expand_ratio_gtu = 1
expand_ratio_glu = 1.5
gtu_use_decay = True
gtu_gamma = [0.95,0.9]
opt_wd = [0.0]
lr={'cifar':[0.00725], 'imdb':[0.0005], 'listops':[0.004], 'pathfinder':[0.0004], 'pathfinderx':[0.0004],'aan':[0.001]}
lr={'cifar':[0.00725], 'imdb':[0.00105], 'listops':[0.00575], 'pathfinder':[0.001], 'pathfinderx':[0.001],'aan':[0.001175]}
opt_wd = {
    'cifar': 0,
    'imdb': 0, 
    'listops': 0, 
    'pathfinder': 0,
    'pathfinderx': 0,
    'aan': 0,
}
# bs = {
#     'cifar':25, 
#     'imdb':16, 
#     'listops':16, 
#     'pathfinder':64, 
#     'pathfinderx':16, 
#     'aan':16
# }
batches = {
    'cifar': [16], 
    'imdb': 32, 
    'listops':128, #32, 
    'pathfinder':128, 
    'pathfinderx':64, 
    'aan': 64,#64, #16
}
# gpus = {
#     'cifar':2, 
#     'imdb':2, 
#     'listops':2, 
#     'pathfinder':2, 
#     'pathfinderx':2, 
#     'aan':1
# }
gpus = {
    'cifar':1, 
    'imdb':2, 
    'listops':2, 
    'pathfinder':4, 
    'pathfinderx':8, 
    'aan':2#,2
}
PREFIX = "/data/qinzhen/code/lra/"

tasks = ["imdb"]
tasks = ["cifar", "imdb","listops","pathfinder","aan"]
archs = ["gtu"]
# tasks = ["cifar", "imdb"]
# tasks = ["listops", "pathfinder", "aan"]
# tasks = ["listops", "pathfinder", "aan"]
# tasks = ["pathfinder", "aan"]
# tasks = ["cifar", "imdb","listops"]
archs = ["tno"]
archs = ["tno2d"]
norm = 'synbatch'
# tasks = ["cifar"]

n_layers = 2
d_model = 64
gtu_dpb_dim = min(d_model//4, 64)
expand_ratio_gtu = 1
expand_ratio_glu = 1.5

# 
n_layers = 4
d_model = 64
gtu_dpb_dim = min(d_model//4, 64)
expand_ratio_gtu = 2
expand_ratio_glu = 2

# n_layers = 2
# d_model = 64
# gtu_dpb_dim = min(d_model//4, 64)
# expand_ratio_gtu = 1.5
# expand_ratio_glu = 1

d_model_dict = {
    'cifar': 32, #64, #[32, 64], #[128, 256, 512],
    'imdb': 128,
    'listops': 128,
    'pathfinder': 64, #[64, 128, 256],
    'pathfinderx':64,
    'aan': 64,
}

n_layers_dict = {
    'cifar': 2, #4, 
    'imdb': [2, 4, 6], 
    'listops': 4, 
    'pathfinder': 6, 
    'pathfinderx':6, 
    'aan': [2, 4], #[2, 4, 6, 8]
}

expand_ratio_gtu_dict = {
    'cifar': 2, 
    'imdb': 1, 
    'listops':1, 
    'pathfinder': 3, 
    'pathfinderx':3, 
    'aan': 1
}

expand_ratio_glu_dict = {
    'cifar': 3, #[1.5, 2], 
    'imdb':2, 
    'listops':1, 
    'pathfinder': 2.5, 
    'pathfinderx':2.5, 
    'aan':1.5
}

# origin
gtu_gamma = {
    'cifar': 1,
    'imdb': 1, 
    'listops': 0.95, 
    'pathfinder': 1,
    'pathfinderx': 1,
    'aan': 0.95,
}

# test
gtu_gamma = {
    'cifar': 0.7, #0.5, #
    'imdb': 0.9, #1, 
    'listops': [1, 0.999], 
    'pathfinder': [1],
    'pathfinderx': [0.9999, 0.99999], #1,
    'aan': [0.8, 0.7, 0.6],
}

norm_dict = {
    'cifar': 'batch', #'synbatch',
    'imdb': 'synbatch', 
    'listops': 'synbatch',
    'pathfinder': 'synbatch',
    'pathfinderx': 'synbatch',
    'aan': 'synbatch',
}

lr_dict = {
    'cifar': [0.007], #0.007, #[1, 0.1, 0.01], #[0.0001], #[0.007, 0.01, 0.001], #[0.00725], 
    'imdb': [0.00105], 
    'listops': 0.0005, #[0.00575], 
    'pathfinder':[0.001], 
    'pathfinderx': [0.0001], #[0.0001],
    'aan': [0.01], #[0.001175]
}

wd_dict = {
    'cifar': [0.001], #0,
    'imdb': [0, 0.001],  #0,
    'listops': 0.1, #0, 
    'pathfinder': 0,
    'pathfinderx': 0,
    'aan': [0.01], #0,
}

dropout_dict = {
    'cifar': 0.1, #[0.1], #0,
    'imdb': [0, 0.1, 0.5, 0.9],  #0,
    'listops': 0, #0, 
    'pathfinder': 0,
    'pathfinderx': 0,
    'aan': 0, #0,
}

dpb_type_dict = {
    'cifar': 1, #0,
    'imdb': 1,  #0,
    'listops': 1, #0, 
    'pathfinder': 1,
    'pathfinderx': 1,
    'aan': 1, #0,
}

dpb_layers_dict = {
    'cifar': 0, #[1, 2, 3], #0,
    'imdb': 3,  #0,
    'listops': 3, #0, 
    'pathfinder': 0, #[1, 2, 3],
    'pathfinderx': 0,
    'aan': 3, #0,
}

gtu_dpb_dim_dict = {
    'cifar': 16, #16, #[8, 16, 32], #0,
    'imdb': 32,  #0,
    'listops': 32, #0, 
    'pathfinder': 16,
    'pathfinderx': 16,
    'aan': 16, #0,
}

prenorm_dict = {
    'cifar': True, #[True, False], 
    'imdb': True, 
    'listops': True, 
    'pathfinder': True,
    'pathfinderx': True,
    'aan': True, 
}

warmup_steps_dict = {
    'cifar': [30000], #30000, #175,
    'imdb': 3000, #80, #
    'listops': [300, 1000, 10000, 20000, 30000], 
    'pathfinder': [312, 5000], #312,
    'pathfinderx': 312,
    'aan': [25000, 30000, 35000, 40000],#800, 
}

tasks = ["aan"]
# tasks = ["pathfinder"]
# tasks = ["imdb"]
# # # tasks = ["pathfinderx"]
# tasks = ["listops"]
# # tasks = ["pathfinder","aan"]
# # tasks = ["imdb","listops","pathfinder","aan"]
tasks = ["cifar"]
# tasks = ["pathfinder", "cifar"]
# tasks = ["pathfinder"]
# tasks = ["pathfinderx"]

archs = ["tno"]
# archs = ["tno2d"]
# archs = ["transnormer"]
archs = ["gtu"]
archs = ["gtu2d"]
# archs = ["transformer_lg"]

def to_iter(*args):
    n = len(args)
    new_args = []
    for i in range(n):
        if not isinstance(args[i], list):
            arg = [args[i]]
        else:
            arg = args[i]
        new_args.append(arg)
    
    return helper(*new_args)

def helper(*args):
    n = len(args)
    if n == 1:
        res = [[arg] for arg in args[0]]
        return res
    else:
        arr = helper(*args[1:])
        res = []
        for par in args[0]:
            for data in arr:
                res.append([par] + list(data))
        return res

for i, task in enumerate(tasks):
    # n_layers = n_layers_dict[task]
    # expand_ratio_gtu = expand_ratio_gtu_dict[task]
    # expand_ratio_glu = expand_ratio_glu_dict[task]
    pars = to_iter(archs, n_layers_dict[task], expand_ratio_gtu_dict[task], expand_ratio_glu_dict[task], d_model_dict[task], gtu_gamma[task], batches[task], norm_dict[task], lr_dict[task], wd_dict[task], dropout_dict[task], dpb_type_dict[task], dpb_layers_dict[task], gtu_dpb_dim_dict[task], prenorm_dict[task], warmup_steps_dict[task])
    print(pars)
    print(task)
    print(len(pars))
    time.sleep(1)
    for arch, n_layers, expand_ratio_gtu, expand_ratio_glu, d_model, gamma, total_batch, norm, lr, wd, dropout, dpb_type, dpb_layers, gtu_dpb_dim, prenorm, warmup_steps in pars:
        # gtu_dpb_dim = min(d_model//4, 32)
        if task == 'imdb':
            seq_len = 4096
            if not gtu_use_decay:
                os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                sys.exit(0)
            else:
                gpu = gpus[task]
                # batch = batches[task] // gpu
                batch = total_batch // gpu
                workers = gpu * 20
                for i in range(1):
                    # lr = round(lr + 0.000025,6)
                    print("imdb lr: ",lr)
                    time.sleep(1)
                    pid = os.fork()
                    if pid == 0:
                        os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                        sys.exit(0)
        elif task == 'cifar':
            seq_len = 1024
            if not gtu_use_decay:
                os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu}')
                sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("cifar lr: ",lr)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    p = os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                    sys.exit(0)
        elif task == 'listops':
            seq_len = 2048
            if not gtu_use_decay:
                os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu}')
                sys.exit(0)
            else:
                gpu = gpus[task]
                # batch = batches[task] // gpu
                batch = total_batch // gpu
                workers = gpu * 20
                print("listops lr: ",lr)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                    sys.exit(0)
        elif task == 'pathfinder':
            seq_len = 1024
            if not gtu_use_decay:
                print("pathfinder lr: ",lr)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                    sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("pathfinder lr: ",lr)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                    sys.exit(0)
        elif task == 'pathfinderx':
            seq_len = 128*128
            if not gtu_use_decay:
                print("pathfinderx lr: ",lr)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu}')
                    sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("pathfinderx lr: ",lr)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                    sys.exit(0)
        elif task == 'aan':
            seq_len = 4000
            if not gtu_use_decay:
                os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("aan lr: ",lr)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}')
                    sys.exit(0)
