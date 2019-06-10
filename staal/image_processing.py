import cv2
import torch 
from torchvision import transforms, utils
import torch

class LoadImageTexture(object):

    def __call__(self, gray):
        gray_torch = torch.from_numpy(gray).unsqueeze(0).float() / 255.0
        gray_torch.requires_grad = False
        return gray_torch
        

def preprocess_image(image):
    preprocess = transforms.Compose([
        LoadImageTexture()
    ])
    return preprocess(image)

def load_image(image_path):
    image = cv2.imread(image_path, 0)
    return preprocess_image(image)
