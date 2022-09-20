import os
import re


# job_ids = ["4296378", "4292997"]

def parse(file, arch, task, job_id, arr):
    res = dict()
    for a in arr:
        res[a] = 0
    res["arch"] = arch
    res["task"] = task
    res["job_id"] = job_id
    try:
        with open(file, encoding="utf-8") as f:
            for line in f.readlines():
                for col in arr:
                    if f" {col}:" in line:
                        data = line.strip().split(':')[-1]
                        res[col] = data
                if "Total params" in line:
                    tmp = line.split()
                    res["Total params"] = tmp[0] + tmp[1]
                if f"final/val/accuracy" in line:
                    p = float(line.strip().replace(',','').split()[1])
                    res["val/accuracy"] = max(res["val/accuracy"], p)
                if f"final/test/accuracy" in line:
                    p = float(line.strip().replace(',','').split()[1])
                    res["test/accuracy"] = max(res["test/accuracy"], p)
        res["total_batch_size"] = str(int(res["gpus"]) * int(res["batch_size"]))

        col = arr[0]
        for a in arr[1:]:
            col += f",{a}"

        string = res[arr[0]]
        for a in arr[1:]:
            string += f",{res[a]}"
        if res["val/accuracy"] != 0:
            print(string)
    except:
        print(res["job_id"])
    
path = "/mnt/lustre/share_data/hanxiaodong/lra_log/log/pathfinderx"

arr = ["arch", "task", "val/accuracy", "test/accuracy", "Total params", "norm", "expand_ratio_tno", "expand_ratio_glu", "tno_use_decay", "tno_gamma", "total_batch_size", "gpus", "batch_size", "d_model", "n_layers", "lr", "weight_decay", "num_warmup_steps", "num_training_steps", "job_id", "tno_type"]
col = arr[0]
for a in arr[1:]:
    col += f",{a}"
print(col)

job_ids = []
for file in os.listdir(path):
    if file.split('.')[-1] == 'log':
        job_id = file.split('-')[1]
        job_ids.append(job_id)

for job_id in job_ids:
    job_id = str(job_id)
    for file in os.listdir(path):
        if job_id in file:
            tmp = file.split('-')[-1].split('_')
            task = tmp[0]
            arch = tmp[1].split('.')[0]
            abs_path = os.path.join(path, file)
            parse(abs_path, arch, task, job_id, arr)
            # # nmt
            #