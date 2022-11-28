import os
import sys
import time
from turtle import pd

EXPAND_RATIO_TNO=1
EXPAND_RATIO_GLU=1

cards=1
tasks = ["imdb","listops","pathfinder","pathfinderx","aan"]
tasks = ['pathfinderx']
archs = ["tno"]
archs = ['tno2d']
bs = {'cifar':25, 'imdb':16, 'listops':16, 'pathfinder':64, 'pathfinderx':16, 'aan':16}
# change
n_layers = 2
d_model = 64
tno_dpb_dim = min(d_model//4, 64)
# change
norm = 'batch'
expand_ratio_tno = 1
expand_ratio_glu = 1.5
tno_use_decay = True
tno_gamma = [0.95,0.9]
opt_wd = [0.0]
lr={'cifar':[0.00725], 'imdb':[0.0005], 'listops':[0.004], 'pathfinder':[0.0004], 'pathfinderx':[0.0004],'aan':[0.001]}
lr={'cifar':[0.00725], 'imdb':[0.00105], 'listops':[0.00575], 'pathfinder':[0.001], 'pathfinderx':[0.001],'aan':[0.001175]}
gtu_gamma = {
    'cifar': 1,
    'imdb': 1, 
    'listops': 0.95, 
    'pathfinder': 1,
    'pathfinderx': 1,
    'aan': 0.95,
}

tasks = ["imdb"]
archs = ["gtu"]


for i, task in enumerate(tasks):
    time.sleep(1)
    if task == 'imdb':
        seq_len = 4096
        if not tno_use_decay:
            os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} 0 {lr[task][0]} {opt_wd[0]} 2')
            sys.exit(0)
        else:
            # for j, gamma in enumerate(tno_gamma):
            #     lr_one = lr[task][0]
            #     for i in range(1):
            #         # lr_one = round(lr_one + 0.000025,6)
            #         print("imdb lr: ",lr_one)
            #         time.sleep(2)
            #         pid = os.fork()
            #         if pid == 0:
            #             os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
            #             sys.exit(0)
            gamma = gtu_gamma[task]
            lr_one = lr[task][0]
            for i in range(1):
                # lr_one = round(lr_one + 0.000025,6)
                print("imdb lr: ",lr_one)
                time.sleep(2)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
                    sys.exit(0)
    if task == 'cifar':
        seq_len = 1024
        if not tno_use_decay:
            os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} 0 {lr[task][0]} {opt_wd[0]} 2')
            sys.exit(0)
        else:
            # for j, gamma in enumerate(tno_gamma):
            #     lr_one = lr[task][0]
            #     for i in range(1):
            #         # lr_one = round(lr_one + 0.00025,5)
            #         print("cifar lr: ",lr_one)
            #         time.sleep(2)
            #         pid = os.fork()
            #         if pid == 0:
            #             os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
            #             sys.exit(0)
            gamma = gtu_gamma[task]
            lr_one = lr[task][0]
            for i in range(1):
                # lr_one = round(lr_one + 0.00025,5)
                print("cifar lr: ",lr_one)
                time.sleep(2)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
                    sys.exit(0)
    if task == 'listops':
        seq_len = 2048
        if not tno_use_decay:
            os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} 0 {lr[task][0]} {opt_wd[0]} 2')
            sys.exit(0)
        else:
            # for j, gamma in enumerate(tno_gamma):
            #     lr_one = lr[task][0]
            #     for i in range(40):
            #         lr_one = round(lr_one + 0.00025,5)
            #         print("listops lr: ",lr_one)
            #         time.sleep(2)
            #         pid = os.fork()
            #         if pid == 0:
            #             os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
            #             sys.exit(0)
            gamma = gtu_gamma[task]
            lr_one = lr[task][0]
            for i in range(40):
                lr_one = round(lr_one + 0.00025,5)
                print("listops lr: ",lr_one)
                time.sleep(2)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
                    sys.exit(0)
    if task == 'pathfinder':
        seq_len = 1024
        if not tno_use_decay:
            lr_one = lr[task][0]
            for i in range(1):
                # lr_one = round(lr_one + 0.000025,6)
                print("pathfinder lr: ",lr_one)
                time.sleep(2)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} 0 {lr_one} {opt_wd[0]} 2')
                    sys.exit(0)
        else:
            # for j, gamma in enumerate(tno_gamma):
            #     for lr_one in lr[task]:
            #         print("pathfinder lr: ",lr_one)
            #         time.sleep(1)
            #         pid = os.fork()
            #         if pid == 0:
            #             os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
            #             sys.exit(0)
            gamma = gtu_gamma[task]
            for lr_one in lr[task]:
                print("pathfinder lr: ",lr_one)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
                    sys.exit(0)
    if task == 'pathfinderx':
        seq_len = 128*128
        if not tno_use_decay:
            lr_one = lr[task][0]
            for i in range(1):
                # lr_one = round(lr_one + 0.00025,5)
                print("pathfinderx lr: ",lr_one)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} 0 {lr_one} {opt_wd[0]} 2')
                    sys.exit(0)
        else:
            # for j, gamma in enumerate(tno_gamma):
            #     for lr_one in lr[task]:
            #         print("pathfinderx lr: ",lr_one)
            #         time.sleep(1)
            #         pid = os.fork()
            #         if pid == 0:
            #             os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
            #             sys.exit(0)
            gamma = gtu_gamma[task]
            for lr_one in lr[task]:
                print("pathfinderx lr: ",lr_one)
                time.sleep(1)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 2')
                    sys.exit(0)
    if task == 'aan':
        seq_len = 4000
        if not tno_use_decay:
            os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} 0 {lr[task][0]} {opt_wd[0]} 1')
            sys.exit(0)
        else:
            # for j, gamma in enumerate(tno_gamma):
            #     lr_one = lr[task][0]
            #     for i in range(50):
            #         lr_one = round(lr_one + 0.000025,6)
            #         print("aan lr: ",lr_one)
            #         time.sleep(2)
            #         pid = os.fork()
            #         if pid == 0:
            #             os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 1')
            #             sys.exit(0)
            gamma = gtu_gamma[task]
            lr_one = lr[task][0]
            for i in range(50):
                lr_one = round(lr_one + 0.000025,6)
                print("aan lr: ",lr_one)
                time.sleep(2)
                pid = os.fork()
                if pid == 0:
                    os.system(f'sh run_task.sh {task} {archs[0]} {bs[task]} {n_layers} {d_model} {tno_dpb_dim} {norm} {expand_ratio_tno} {expand_ratio_glu} {tno_use_decay} {gamma} {lr_one} {opt_wd[0]} 1')
                    sys.exit(0)
