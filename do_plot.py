import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import math

plt.style.use('seaborn-whitegrid')



def draw(root_path, color):
    fig, ax = plt.subplots()

    mem_log_file = open("debug/mem.log")
    acc_log_file = open("debug/acc.log")
    speed_log_file = open("debug/speed.log")

    plt.xlabel("Speed (steps per sec)", fontsize=30)
    plt.ylabel("Long-Range Arena Score", fontsize=30)
    plt.tick_params(labelsize=22)
    
    for mem_line, acc_line, speed_line in zip(mem_log_file, acc_log_file, speed_log_file):
        mem_meta = mem_line.split()
        acc_meta = acc_line.split()
        speed_meta = speed_line.split()

        print('Read ' + mem_meta[0])

        mem = float(mem_meta[-1][:-3])
        if mem_meta[-1][-3:] == "GiB":
            mem = mem * 1024

        acc = float(acc_meta[-1])
        speed = float(speed_meta[-1])

        print("mem: " + mem_meta[-1])
        print("speed: " + speed_meta[-1])
        print("acc: " + acc_meta[-1])

        name = mem_meta[0][0:-4]
        if name == "relu_multi":
            name = "baseline"
            continue
        elif name == "relu_weight_multi":
            name = "cosFormer"
        
        if name == "bigbird":
            name = "BigBird"
            annotate_xy = (speed + 1.6, acc + 0.4)
        elif name == "transformer":
            name = "Transformer"
            annotate_xy = (speed + 2.4 , acc - 0.1)
        elif name == "longformer":
            name = "Longformer"
            annotate_xy = (speed + 2.4, acc - 0.3)
        elif name == "synthesizer":
            name = "Synthesizer"
            annotate_xy = (speed + 2.5, acc - 0.6)
        elif name == "sparse_transformer":
            name = "Sparse Transformer"
            annotate_xy = (speed - 2.8, acc - 1.2 )
        elif name == "local":
            name = "Local attention"
            annotate_xy = (speed + 1.5, acc - 0.4)
        elif name == "reformer":
            name = "Reformer"
            annotate_xy = (speed + 1.5, acc - 0.4)
        elif name == "sinkhorn_transformer":
            name = "Sinkhorn Transformer"
            annotate_xy = (speed + 1.5, acc - 0.1)
        elif name == "cosFormer":
            annotate_xy = (speed + 1.5, acc - 0.4)
        elif name == "linear_transformer":
            name = "Linear Transformer"
            annotate_xy = (speed - 4.6, acc - 0.9)
        elif name == "linformer":
            name = "Linformer"
            annotate_xy = (speed - 3, acc + 0.6)
        elif name == "performer":
            name = "Performer"
            annotate_xy = (speed - 3, acc - 0.9)
        else:
            annotate_xy = (speed + 1 * mem / 10240.0, acc + 1 * mem / 10240.0)

        if name == 'BigBird':
            ax.scatter([speed], [acc], alpha=0.6, s = 1500 * mem / 1024.0, linewidths=2, color=(0.1,0.1,0.8))
        else:
            ax.scatter([speed], [acc], alpha=0.6, s = 1500 * mem / 1024.0, linewidths=2)

        if name == 'cosFormer':
            ax.annotate(name, annotate_xy, fontsize=26, weight='bold', color=(0.8,0.2,0.2),fontstyle='oblique',family='Verdana')
        else:
            ax.annotate(name, annotate_xy, fontsize=24)
        ax.spines['bottom'].set_linewidth(3);###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(3);####设置左边坐标轴的粗细

    print('Draw')
    plt.show()

draw("sd","s")
