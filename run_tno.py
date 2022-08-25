import os
import time
import sys


EXPAND_RATIO_TNO=1
EXPAND_RATIO_GLU=1

cards=1
tasks = ["imdb","cifar","listops","pathfinder","aan"]
# tasks = ['cifar']
archs = ["tno"]
bs = {'cifar':256, 'imdb':16, 'listops':16, 'pathfinder':64, 'aan':16}
n_layers = 2
d_model = 64
tno_dpb_dim = min(d_model//4, 64)
norm = 'batch'
expand_ratio_tno = 1
expand_ratio_glu = 1
tno_use_decay = False
tno_gamma = 0.999
for i, task in enumerate(tasks):
    time.sleep(5)
    pid = os.fork()
    if pid == 0:
        if task == 'imdb':
            seq_len = 4096
            if not tno_use_decay:
                os.system(f'sh run_tno.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {tno_gamma} 2')
            sys.exit(0)
        if task == 'cifar':
            seq_len = 1024
            if not tno_use_decay:
                os.system(f'sh run_tno.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {tno_gamma} 2')
            sys.exit(0)
        if task == 'listops':
            seq_len = 2048
            if not tno_use_decay:
                os.system(f'sh run_tno.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {tno_gamma} 2')
            sys.exit(0)
        if task == 'pathfinder':
            seq_len = 1024
            if not tno_use_decay:
                os.system(f'sh run_tno.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {tno_gamma} 2')
            sys.exit(0)
        if task == 'aan':
            seq_len = 4000
            if not tno_use_decay:
                os.system(f'sh run_tno.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {tno_gamma} 1')
            sys.exit(0)