import numpy as np
import cv2
from staal.pretrained_resnet import PretrainedResnet
import torch 
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, random, string
from torch.autograd import Variable
import sys
import numpy as np
import cv2
import time
import torchvision.models as models
import torch.autograd as autograd
import torch.nn as nn
import torch
import torch.nn.functional as F

def generate_dataset(root, filename):
    cap = cv2.VideoCapture(f'{root}/{filename}')
    frame_index = 0
    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            frame_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_cv = contrast_equalization.apply(frame_cv)
            cv2.imshow('frame', frame_cv)
            base_filename = os.path.splitext(filename)[0]
            folder_name = f"data/image-sequences/{base_filename}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            cv2.imwrite(f"{folder_name}/{frame_index}.png", frame_cv)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('p'):
                time.sleep(100)
            frame_index = frame_index + 1
        except:
            return None
    cap.release()
    cv2.destroyAllWindows()

contrast_equalization = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
for root, dirs, files in os.walk("data/videos/"):
    for filname in files:
        if filname.endswith('.webm'):
            print(f"Generating dataset for file {filname}")
            generate_dataset(root, filname)
