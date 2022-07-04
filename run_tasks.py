import os
import time
import sys

tasks = ["imdb","cifar","listops","pathfinder","aan"]
archs = ["flash","flash_linear","lg","ls","performer"]
archs = ["lg"]

flash_args={'cifar':[1024,128],"imdb":[4096,128],"listops":[2048,128],"pathfinder":[1024,128],"aan":[4000,128]}
flash_linear_args={'cifar':[1024,128],"imdb":[4096,128],"listops":[2048,128],"pathfinder":[1024,128],"aan":[4000,128]}
lg_args=[8,8,64]
ls_args={'cifar':[8,8,1024],"imdb":[8,8,4096],"listops":[8,8,2048],"pathfinder":[8,8,1024],"aan":[8,8,4000]}
performer_args=[8,32]

### new config
flash_args={'cifar':[1024,64],"imdb":[4096,64],"listops":[2048,64],"pathfinder":[1024,64],"aan":[4000,64]}
flash_linear_args={'cifar':[1024,64],"imdb":[4096,64],"listops":[2048,64],"pathfinder":[1024,64],"aan":[4000,64]}
ls_args={'cifar':[2,8,1024],"imdb":[2,8,4096],"listops":[2,8,2048],"pathfinder":[2,8,1024],"aan":[2,8,4000]}
lg_args=[2,2,64]
cosformer_args={'cifar':[8,1024],"imdb":[8,4096],"listops":[8,2048],"pathfinder":[8,1024],"aan":[8,4000]}  # heads, max_length
# n_layers, d_model
arch_args={'cifar':
                {"s4":{"flash":[11,512],"flash_linear":[11,512],"lg":[6,512],"ls":[6,512]},
                # "lra":{"flash":[6,64],"flash_linear":[6,64],"lg":[4,64],"ls":[4,64]}},
                "lra":{"flash":[2,64],"flash_linear":[2,64],"lg":[2,64],"ls":[2,64],"cosformer":[2,64]}},
            'listops':
                {"s4":{"flash":[10,128],"flash_linear":[10,128],"lg":[6,128],"ls":[6,128]},
                # "lra":{"flash":[11,512],"flash_linear":[11,512],"lg":[6,512],"ls":[6,512]}},
                "lra":{"flash":[2,64],"flash_linear":[2,64],"lg":[2,64],"ls":[2,64],"cosformer":[2,64]}},
            'pathfinder':
                {"s4":{"flash":[10,256],"flash_linear":[10,256],"lg":[6,256],"ls":[6,256]},
                # "lra":{"flash":[6,128],"flash_linear":[6,128],"lg":[6,128],"ls":[6,128]}},
                "lra":{"flash":[2,64],"flash_linear":[2,64],"lg":[2,64],"ls":[2,64],"cosformer":[2,64]}},
            'imdb':
                {"s4":{"flash":[5,64],"flash_linear":[5,64],"lg":[4,64],"ls":[4,64]},
                # "lra":{"flash":[11,512],"flash_linear":[11,512],"lg":[6,512],"ls":[6,512]}},
                "lra":{"flash":[2,64],"flash_linear":[2,64],"lg":[2,64],"ls":[2,64],"cosformer":[2,64]}},
            'aan':
                {"s4":{"flash":[10,256],"flash_linear":[10,256],"lg":[6,256],"ls":[6,256]},
                # "lra":{"flash":[6,128],"flash_linear":[6,128],"lg":[4,128],"ls":[4,128]}}
                "lra":{"flash":[2,64],"flash_linear":[2,64],"lg":[2,64],"ls":[2,64],"cosformer":[2,64]}}
                }

norm='batch'
tasks_tmp = ['cifar']
archs_tmp = ["flash","ls"]

tasks_tmp = ["imdb","cifar","aan", "listops","pathfinder"]
tasks_tmp = ["aan"]
tasks_tmp = ["imdb","cifar", "listops","pathfinder"]
# archs_tmp = ["lg"]
tasks_tmp = ["imdb","cifar", "listops","pathfinder"]
archs_tmp = ["lg", "flash","flash_linear","ls"]
tasks_tmp = ["aan"]
archs_tmp = ["cosformer"]
for j, task in enumerate(tasks_tmp):
    for i, arch in enumerate(archs_tmp):
        for norm in ['batch']:
            if "lg" in arch:
                tmp = [(True, "1+elu"), (False, "elu")]
            else:
                tmp = [(True, "1+elu")]
            print(task, arch, tmp)
            for use_softmax, act_fun in tmp:
                print(use_softmax, act_fun)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    name = f"{arch}_{task}"
                    print(name)
                    # TODO check imdb task
                    if task == 'imdb':
                        seq_len = 4096
                        lg_args=[2,2,64]
                        args=flash_args[task]
                        if arch == 'flash':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} {flash_args[task][0]} {flash_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'flash_linear':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 {flash_linear_args[task][0]} {flash_linear_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'lg':
                            # os.system(f'sh run_task.sh {task} {arch} {5} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 {lg_args[0]} {lg_args[1]} {lg_args[2]} 0 0 0 0 0 8')
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 {lg_args[0]} {lg_args[1]} {lg_args[2]} 0 0 0 0 0 {use_softmax} {act_fun} 0 0 2')
                        if arch == 'ls':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 {ls_args[task][0]} {ls_args[task][1]} {ls_args[task][2]} 0 0 0 0 0 0 2')
                        if arch == 'cosformer':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 {cosformer_args[task][0]} {cosformer_args[task][1]} 2')
                        sys.exit(0)
                    # TODO check cifar task
                    if task == 'cifar':
                        seq_len = 1024
                        lg_args=[2,2,64]
                        if arch == 'flash':
                            os.system(f'sh run_task.sh {task} {arch} {256} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} {flash_args[task][0]} {flash_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'flash_linear':
                            os.system(f'sh run_task.sh {task} {arch} {256} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 {flash_linear_args[task][0]} {flash_linear_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'lg':
                            os.system(f'sh run_task.sh {task} {arch} {256} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 {lg_args[0]} {lg_args[1]} {lg_args[2]} 0 0 0 0 0 {use_softmax} {act_fun} 0 0 2')
                        if arch == 'ls':
                            os.system(f'sh run_task.sh {task} {arch} {256} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 {ls_args[task][0]} {ls_args[task][1]} {ls_args[task][2]} 0 0 0 0 0 02')
                        if arch == 'cosformer':
                            os.system(f'sh run_task.sh {task} {arch} {256} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 {cosformer_args[task][0]} {cosformer_args[task][1]} 2')                        
                        sys.exit(0)
                        # if arch == 'performer':
                        #     os.system(f'sh run_task.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}')
                    # TODO check listops task
                    if task == 'listops':
                        seq_len = 2048
                        if arch == 'flash':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} {flash_args[task][0]} {flash_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'flash_linear':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 {flash_linear_args[task][0]} {flash_linear_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'lg':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 {lg_args[0]} {lg_args[1]} {lg_args[2]} 0 0 0 0 0 {use_softmax} {act_fun} 0 0 2')
                        if arch == 'ls':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 {ls_args[task][0]} {ls_args[task][1]} {ls_args[task][2]} 0 0 0 0 0 0 2')
                        if arch == 'cosformer':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 {cosformer_args[task][0]} {cosformer_args[task][1]} 2')
                        sys.exit(0)
                        # if arch == 'performer':
                        #     os.system(f'sh run_task.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}')
                    # TODO check pathfinder task
                    if task == 'pathfinder':
                        seq_len = 1024
                        if arch == 'flash':
                            os.system(f'sh run_task.sh {task} {arch} {64} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} {flash_args[task][0]} {flash_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'flash_linear':
                            os.system(f'sh run_task.sh {task} {arch} {64} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 {flash_linear_args[task][0]} {flash_linear_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 2')
                        if arch == 'lg':
                            os.system(f'sh run_task.sh {task} {arch} {64} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 {lg_args[0]} {lg_args[1]} {lg_args[2]} 0 0 0 0 0 {use_softmax} {act_fun} 0 0 2')
                        if arch == 'ls':
                            os.system(f'sh run_task.sh {task} {arch} {64} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 {ls_args[task][0]} {ls_args[task][1]} {ls_args[task][2]} 0 0 0 0 0 0 2')
                        if arch == 'cosformer':
                            os.system(f'sh run_task.sh {task} {arch} {64} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 {cosformer_args[task][0]} {cosformer_args[task][1]} 2')
                        sys.exit(0)
                    # TODO check aan task
                    if task == 'aan':
                        seq_len = 4000
                        lg_args=[2,2,64]
                        if arch == 'flash':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} {flash_args[task][0]} {flash_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1')
                        if arch == 'flash_linear':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 {flash_linear_args[task][0]} {flash_linear_args[task][1]} 0 0 0 0 0 0 0 0 0 0 0 0 1')
                        if arch == 'lg':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 {lg_args[0]} {lg_args[1]} {lg_args[2]} 0 0 0 0 0 8 {use_softmax} {act_fun} 0 0 1')
                        if arch == 'ls':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 {ls_args[task][0]} {ls_args[task][1]} {ls_args[task][2]} 0 0 0 0 0 0 1')
                        if arch == 'cosformer':
                            os.system(f'sh run_task.sh {task} {arch} {16} {arch_args[task]["lra"][arch][0]} {arch_args[task]["lra"][arch][1]} {norm} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 {cosformer_args[task][0]} {cosformer_args[task][1]} 1')                         
                        sys.exit(0)
                    # os.system(f'sh /mnt/lustre/qinzhen/experiment/nlp/roberta_m3t.sh 8 {arch} {name} 0.0005 0.0 MMG')
                    sys.exit(0)
