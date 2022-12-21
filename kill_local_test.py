import os
import signal

os.system(f'ps aux | grep -v PID | grep lra | grep batch  > tmp')

with open("tmp") as f:
    for i, text in enumerate(f):
        if i > 0:
            pid = int(text.strip().split()[1])
            print(i, pid, text,)
            os.kill(pid, signal.SIGINT)

os.remove("tmp")