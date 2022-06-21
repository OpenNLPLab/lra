import os
import time
import sys

tasks = ["imdb","cifar","listops","pathfinder","aan"]
archs = ["flash","flash-linear","lg","ls","performer"]
archs = ["lg"]

flash_args={'cifar':[1024,128],"imdb":[5000,128],"listops":[2048,128],"pathfinder":[1024,128],"aan":[4000,128]}
flash_linear_args={'cifar':[1024,128],"imdb":[5000,128],"listops":[2048,128],"pathfinder":[1024,128],"aan":[4000,128]}
lg_args=[8,8,32]
ls_args={'cifar':[8,8,1024],"imdb":[8,8,5000],"listops":[8,8,2048],"pathfinder":[8,8,1024],"aan":[8,8,4000]}
performer_args=[8,32]

### new config
# flash_args={'cifar':[1024,64],"imdb":[4096,64],"listops":[2048,64],"pathfinder":[1024,64],"aan":[4000,64]}
# flash_linear_args={'cifar':[1024,64],"imdb":[4096,64],"listops":[2048,64],"pathfinder":[1024,64],"aan":[4000,64]}
# ls_args={'cifar':[2,8,1024],"imdb":[2,8,4096],"listops":[2,8,2048],"pathfinder":[2,8,1024],"aan":[2,8,4000]}
# lg_args=[2,2,64]

# n_layers, d_model
arch_args={'cifar':
                {"s4":{"flash":[11,512],"flash-linear":[11,512],"lg":[6,512],"ls":[6,512]},
                # "lra":{"flash":[6,64],"flash-linear":[6,64],"lg":[4,64],"ls":[4,64]}},
                "lra":{"flash":[2,64],"flash-linear":[2,64],"lg":[2,64],"ls":[2,64]}},
            'listops':
                {"s4":{"flash":[10,128],"flash-linear":[10,128],"lg":[6,128],"ls":[6,128]},
                # "lra":{"flash":[11,512],"flash-linear":[11,512],"lg":[6,512],"ls":[6,512]}},
                "lra":{"flash":[2,64],"flash-linear":[2,64],"lg":[2,64],"ls":[2,64]}},
            'pathfinder':
                {"s4":{"flash":[10,256],"flash-linear":[10,256],"lg":[6,256],"ls":[6,256]},
                # "lra":{"flash":[6,128],"flash-linear":[6,128],"lg":[6,128],"ls":[6,128]}},
                "lra":{"flash":[2,64],"flash-linear":[2,64],"lg":[2,64],"ls":[2,64]}},
            'imdb':
                # {"s4":{"flash":[5,64],"flash-linear":[5,64],"lg":[4,64],"ls":[4,64], "performer":[4,64],"vanilla":[4,64]},
                {"s4":{"flash":[6,128],"flash-linear":[6,128],"lg":[6,128],"ls":[6,128], "performer":[6,128],"vanilla":[6,128]},
                # "lra":{"flash":[11,512],"flash-linear":[11,512],"lg":[6,512],"ls":[6,512]}},
                "lra":{"flash":[2,64],"flash-linear":[2,64],"lg":[2,64],"ls":[2,64],"performer":[2,64]},"vanilla":[2,64]},
            'aan':
                {"s4":{"flash":[10,256],"flash-linear":[10,256],"lg":[6,256],"ls":[6,256]},
                # "lra":{"flash":[6,128],"flash-linear":[6,128],"lg":[4,128],"ls":[4,128]}}
                "lra":{"flash":[2,64],"flash-linear":[2,64],"lg":[2,64],"ls":[2,64]}}
                }

seq_len_args=[2000]



norm='batch'
tasks_tmp = ['cifar']
archs_tmp = ["flash","ls"]

tasks_tmp = ["imdb","cifar","aan", "listops","pathfinder"]
tasks_tmp = ["aan"]
tasks_tmp = ["imdb","cifar", "listops","pathfinder"]
# archs_tmp = ["lg"]
tasks_tmp = ["imdb","cifar", "listops","pathfinder"]
archs_tmp = ["lg", "flash","flash-linear","ls"]

tasks_tmp = ["imdb"]
archs_tmp = ["lg"]
for j, task in enumerate(tasks_tmp):
    for i, arch in enumerate(archs_tmp):
        for norm in ['batch']:
            # if "lg" in arch:
            #     tmp = [(True, "1+elu"), (False, "elu")]
            # else:
            tmp = [(False, "elu")]
            print(arch)
            for use_softmax, act_fun in tmp:
                pid = os.fork()
                if pid == 0:
                    # TODO check imdb task
                    if task == 'imdb':
                        for seq_len in seq_len_args:
                            # args=flash_args[task]
                            if arch == 'flash':
                                os.system(f'bash speed_test.sh {task} {arch} {20} {arch_args[task]["s4"][arch][0]} {arch_args[task]["s4"][arch][1]} {norm} {flash_args[task][0]} {flash_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 1 {seq_len}')
                            if arch == 'flash-linear':
                                os.system(f'bash speed_test.sh {task} {arch} {20} {arch_args[task]["s4"][arch][0]} {arch_args[task]["s4"][arch][1]} {norm} 0 0 {flash_linear_args[task][0]} {flash_linear_args[task][1]} 0 0 0 0 0 0 0 0 0 0 1 {seq_len}')
                            if arch == 'lg':
                                os.system(f'bash speed_test.sh {task} {arch} {20} {arch_args[task]["s4"][arch][0]} {arch_args[task]["s4"][arch][1]} {norm} 0 0 0 0 {lg_args[0]} {lg_args[1]} {lg_args[2]} 0 0 0 0 0 {use_softmax} {act_fun} 1 {seq_len}')
                            if arch == 'ls':
                                os.system(f'bash speed_test.sh {task} {arch} {20} {arch_args[task]["s4"][arch][0]} {arch_args[task]["s4"][arch][1]} {norm} 0 0 0 0 0 0 0 {ls_args[task][0]} {ls_args[task][1]} {ls_args[task][2]} 0 0 0 0 1 {seq_len}')
                            if arch == 'performer':
                                os.system(f'bash speed_test.sh {task} {arch} {20} {arch_args[task]["s4"][arch][0]} {arch_args[task]["s4"][arch][1]} {norm} 0 0 0 0 0 0 0 0 0 0 {performer_args[0]} {performer_args[1]} 0 0 1 {seq_len}')
                            if arch == 'vanilla':
                                os.system(f'bash speed_test.sh {task} {arch} {20} {arch_args[task]["s4"][arch][0]} {arch_args[task]["s4"][arch][1]} {norm} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 {seq_len}')
                            sys.exit(0)
