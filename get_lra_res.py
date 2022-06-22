import os
import re


# job_ids = ["4296378", "4292997"]

def parse(file, arch, task, job_id, arr):
    res = dict()
    # arr = ["arch", "task", "total_batch_size", "gpus", "batch_size", "n_heads", "d_model", "n_layers", "l_max", "lr", "weight_decay", "num_warmup_steps", "num_training_steps", "act_fun", "use_softmax", "job_id"]
    # acc = ["val/accuracy", "test/accuracy"]
    # for a in acc:
    #     res[a] = 0
    # for a in arr:
    #     res[a] = "0"
    for a in arr:
        res[a] = 0
    res["arch"] = arch
    res["task"] = task
    res["job_id"] = job_id

    with open(file, encoding="utf-8") as f:
        for line in f.readlines():
            for col in arr:
                if f" {col}:" in line:
                    data = line.strip().split(':')[-1]
                    res[col] = data
            # for col in acc:
            #     if f"{col}=" in line:
            #         matchObj = re.match(r"(.*)col='(.*?)',", line)
            #         if matchObj:
            #             print(matchObj.group(2))
            #             return
            #             res[col] = matchObj.group(2)
            #     if f"final/{col}" in line:
            #         print(col)
            #         print(line)
            #         return
            if "Total params" in line:
                tmp = line.split()
                res["Total params"] = tmp[0] + tmp[1]
            if f"val/accuracy=" in line:
                matchObj = re.match(r"(.*)val/accuracy=(.*?),", line)
                p = float(matchObj.group(2))
                if matchObj:
                    res["val/accuracy"] = max(res["val/accuracy"], p)
            if f"final/val/accuracy" in line:
                p = float(line.strip().split()[3])
                res["val/accuracy"] = max(res["val/accuracy"], p)

            if f"test/accuracy=" in line:
                matchObj = re.match(r"(.*)test/accuracy=(.*?),", line)
                p = float(matchObj.group(2))
                if matchObj:
                    res["test/accuracy"] = max(res["test/accuracy"], p)
            if f"final/test/accuracy" in line:
                t1 = line.strip().split()
                t2 = [a for a in t1 if '│' not in a]
                p = float(t2[1])
                res["test/accuracy"] = max(res["test/accuracy"], p)
    res["total_batch_size"] = str(int(res["gpus"]) * int(res["batch_size"]))

    # arr = arr[:2] + acc + arr[2:]
    col = arr[0]
    for a in arr[1:]:
        col += f",{a}"
    # print(col)

    string = res[arr[0]]
    for a in arr[1:]:
        string += f",{res[a]}"
    if res["val/accuracy"] != 0:
        print(string)

# job_ids = [4754117, 4754118, 4754119, 4754120, 4754121, 4754122, 4754123, 4754124, 4754125, 4754126, 4754128, 4754130, 4754131, 4754132, 4754133, 4754134]
path = "/mnt/cache/hanxiaodong/lra/arun_log"

arr = ["arch", "task", "val/accuracy", "test/accuracy", "Total params", "norm", "act_fun", "use_softmax", "total_batch_size", "gpus", "batch_size", "n_heads", "d_model", "n_layers", "l_max", "lr", "weight_decay", "num_warmup_steps", "num_training_steps", "job_id"]
col = arr[0]
for a in arr[1:]:
    col += f",{a}"
print(col)

job_ids = []
for file in os.listdir(path):
    if file.split('.')[-1] == 'log':
        job_id = file.split('-')[1]
        job_ids.append(job_id)
import pdb;pdb.set_trace()
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