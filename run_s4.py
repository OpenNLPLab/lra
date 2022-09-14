import os
import time
import sys

tasks = ["imdb","cifar","listops","pathfinder","aan"]
tasks = ['imdb']
arch = 's4'

for i, task in enumerate(tasks):
    time.sleep(5)
    pid = os.fork()
    if pid == 0:
        if task == 'imdb':
            os.system(f'sh run_s4.sh {task} {arch} 2')
        if task == 'cifar':
            os.system(f'sh run_s4.sh {task} {arch} 2')
        if task == 'listops':
            os.system(f'sh run_s4.sh {task} {arch} 2')
        if task == 'pathfinder':
            os.system(f'sh run_s4.sh {task} {arch} 2')
        if task == 'aan':
            os.system(f'sh run_s4.sh {task} {arch} 1')
        sys.exit(0)
