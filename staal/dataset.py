from torch.utils.data import Dataset, DataLoader
import os, random
import torch.nn as nn
import torch
import torch.nn.functional as F
from staal.image_processing import load_image
from torch.autograd import Variable

class ImageDataset(Dataset):

    def __init__(self, width=256, height=256, visual_logging=False):
        self.width  = width
        self.height = height
        self.videos = []
        self.file_to_images = {}
        for root, dirs, _ in os.walk("data/image-sequences"):
            for directory in dirs:
                for inner_root, _, files in os.walk(os.path.join(root, directory)):
                    self.videos.append(directory)
                    self.file_to_images[directory] = [filename for filename in files]
                    self.file_to_images[directory].sort(key=lambda filename: int(os.path.splitext(filename)[0]))
                    self.file_to_images[directory] = [os.path.join(root, os.path.join(directory, filename)) for filename in self.file_to_images[directory]]
 
    def __len__(self):
        return sum([len(self.file_to_images[filename]) for filename in self.file_to_images])

    def __getitem__(self, index):
        video_index = torch.LongTensor(1).random_(0, len(self.videos)).data[0].item()
        video = self.videos[video_index]
        frames = self.file_to_images[video]
        frame_index = ( ( (len(frames) / 2 ) * torch.randn(1) ) / 3).long() + int(len(frames) / 2 )
        if frame_index < 0:
            frame_index = 0
        if frame_index >= len(frames):
            frame_index = len(frames) - 1
        image_path = frames[frame_index]
        image_crop = torch.zeros([1, self.height, self.width])
        image = load_image(image_path)
        #example of image size: torch.Size([1, 280, 1440])
        height = image.size()[1]
        height_offset = torch.LongTensor(1).random_(0, height - self.height - 1)
        width = image.size()[2]
        width_offset = torch.LongTensor(1).random_(0, width - self.width - 1)
        image_crop[:, :, :] = image[ :, height_offset : height_offset + self.height, width_offset : width_offset + self.width]
        image_crop = nn.AdaptiveMaxPool2d((128, 128))(image_crop).data
        return image_crop


class SegmentationDataset(Dataset):

    def __init__(self, width=512, height=256, visual_logging=False):
        self.width  = width
        self.height = height
        self.videos = []
        self.video_to_images = {}
        for root, dirs, _ in os.walk("data/supervised"):
            for directory in dirs:
                for inner_root, _, files in os.walk(os.path.join(root, directory)):
                    self.videos.append(directory)
                    self.video_to_images[directory] = [filename for filename in files]
                    self.video_to_images[directory].sort(key=lambda filename: int(os.path.splitext(filename)[0]))
        
        self.unsupervised_videos = []
        self.unsupervised_to_images = {}
        for root, dirs, _ in os.walk("data/image-sequences"):
            for directory in dirs:
                for inner_root, _, files in os.walk(os.path.join(root, directory)):
                    self.unsupervised_videos.append(directory)
                    self.unsupervised_to_images[directory] = [filename for filename in files]
                    self.unsupervised_to_images[directory].sort(key=lambda filename: int(os.path.splitext(filename)[0]))
                    self.unsupervised_to_images[directory] = [os.path.join(root, os.path.join(directory, filename)) \
                        for filename in self.unsupervised_to_images[directory]]
 
    def __len__(self):
        return 32

    def supervised_sample(self, index):
        video_index = torch.LongTensor(1).random_(0, len(self.videos)).data[0].item()
        video = self.videos[video_index]
        frames = self.video_to_images[video]
        frame_index = torch.LongTensor(1).random_(0, len(self.video_to_images[video])).data[0].item()
        filename = frames[frame_index]
        image_crop = torch.zeros([1, self.height, self.width])
        target_crop = torch.zeros([1, self.height, self.width])
        target = load_image(os.path.join("data/supervised", os.path.join(video, filename)))
        image = load_image(os.path.join("data/image-sequences", os.path.join(video, filename)))
        height = image.size()[1]
        height_offset = torch.LongTensor(1).random_(0, height - self.height - 1)
        width = image.size()[2]
        width_offset = torch.LongTensor(1).random_(0, width - self.width - 1)
        image_crop[:, :, :] = image[ :, height_offset : height_offset + self.height, width_offset : width_offset + self.width]
        target_crop[:, :, :] = target[ :, height_offset : height_offset + self.height, width_offset : width_offset + self.width]
        image_crop = nn.AdaptiveMaxPool2d((128, 256))(image_crop).data
        target_crop = nn.AdaptiveMaxPool2d((128, 256))(target_crop).data
        return image_crop, target_crop

    def __getitem__(self, index):
        if torch.FloatTensor(1).random_(0, 10).data[0].item() > 6:
            return self.supervised_sample(index)
        else:
            video_index = torch.LongTensor(1).random_(0, len(self.unsupervised_videos)).data[0].item()
            video = self.unsupervised_videos[video_index]
            frames = self.unsupervised_to_images[video]
            frame_index = None
            if torch.FloatTensor(1).random_(0, 10).data[0].item() > 5:
                frame_index = torch.LongTensor(1).random_(0, 10).data[0].item()
            else:
                frame_index = len(frames) - torch.LongTensor(1).random_(0, 10).data[0].item() - 1
            filename = frames[frame_index]
            image_crop = torch.zeros([1, self.height, self.width])
            target_crop = torch.zeros([1, 128, 256])
            image = load_image(filename)
            height = image.size()[1]
            height_offset = torch.LongTensor(1).random_(0, height - self.height - 1)
            width = image.size()[2]
            width_offset = torch.LongTensor(1).random_(0, width - self.width - 1)
            image_crop[:, :, :] = image[ :, height_offset : height_offset + self.height, width_offset : width_offset + self.width]
            image_crop = nn.AdaptiveMaxPool2d((128, 256))(image_crop).data
            return image_crop, target_crop




class SegmentationSequenceDataset(Dataset):

    def __init__(self, width=256+128, height=256+128, visual_logging=False):
        self.width  = width
        self.height = height
        self.unsupervised_videos = []
        self.unsupervised_to_images = {}
        for root, dirs, _ in os.walk("data/image-sequences"):
            for directory in dirs:
                for inner_root, _, files in os.walk(os.path.join(root, directory)):
                    self.unsupervised_videos.append(directory)
                    self.unsupervised_to_images[directory] = [filename for filename in files]
                    self.unsupervised_to_images[directory].sort(key=lambda filename: int(os.path.splitext(filename)[0]))
                    self.unsupervised_to_images[directory] = [os.path.join(root, os.path.join(directory, filename)) \
                        for filename in self.unsupervised_to_images[directory]]
        self.video_index = torch.LongTensor(1).random_(0, len(self.unsupervised_videos)).data[0].item()
        self.frames = self.unsupervised_to_images[self.unsupervised_videos[self.video_index]]
 
    def __len__(self):
        return 32

    def __getitem__(self, index):
        frame_index = ( ( (len(self.frames) / 2 ) * torch.randn(1) ) / 3).long() + int(len(self.frames) / 2 ) - 1
        if frame_index < 0:
            frame_index = 0
        if frame_index >= len(self.frames) - 4:
            frame_index = len(self.frames) - 4
        filename = self.frames[frame_index]
        filename_next = self.frames[frame_index + 3]
        image = load_image(filename)
        image_next = load_image(filename_next)
        height = int(image.size()[1] / 2)
        width  = int(image.size()[2] / 2)
        image = nn.AdaptiveMaxPool2d((height, width))(image).data
        image_next = nn.AdaptiveMaxPool2d((height, width))(image_next).data
        return image, image_next