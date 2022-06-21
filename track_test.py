import torch
import inspect

from torchvision import models
from gpu_mem_track import  MemTracker

device = torch.device('cuda:0')

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker

gpu_tracker.track()                     # run function between the code line where uses GPU
cnn = models.vgg19(pretrained=True).features.to(device).eval()
gpu_tracker.track()                     # run function between the code line where uses GPU

dummy_tensor_1 = torch.randn(30, 3, 512, 512).float().to(device)  # 30*3*512*512*4/1000/1000 = 94.37M
dummy_tensor_2 = torch.randn(40, 3, 512, 512).float().to(device)  # 40*3*512*512*4/1000/1000 = 125.82M
dummy_tensor_3 = torch.randn(60, 3, 512, 512).float().to(device)  # 60*3*512*512*4/1000/1000 = 188.74M

gpu_tracker.track()

