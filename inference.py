from staal.model import TextureFinder
from staal.image_processing import preprocess_image
import torch
import torch.nn as nn
import cv2
import traceback
import numpy as np
import argparse

texture_finder = torch.load("models/model.model").cuda().eval()
texture_segment = torch.load("models/texture-segment.model").cuda().eval()

def segment_frame(frame):
  frame = preprocess_image(frame).cuda()
  height = int(frame.size()[1] / 2)
  width  = int(frame.size()[2] / 2)
  frame = nn.AdaptiveMaxPool2d((height, width))(frame).data
  segmentations = []
  reconstructed, mus, log_var, samples, fine_embeddings, coarse_embeddings = texture_finder(frame.unsqueeze(0))
  segmentations.append(samples)
  for i in range(5):
    segmentations.append(texture_finder.sample_from_mu_log_var(mus, log_var))
  segment = texture_segment(torch.cat(segmentations, 0))
  return segment.mean(0).unsqueeze(0), frame.unsqueeze(0), reconstructed

def inference_video(video_path, show_reconstruction):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            frame_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_cv = contrast_equalization.apply(frame_cv)
            segmented_frame, frame_pytorch, reconstructed = segment_frame(frame_cv)
            height_frame = frame_pytorch.size()[2]
            width_frame = frame_pytorch.size()[3]
            height_segment = segmented_frame.size()[2]
            width_segment = segmented_frame.size()[3]
            if height_frame != height_segment or width_frame != width_segment:
               segmented_frame = nn.AdaptiveMaxPool2d((height_frame, width_frame))(segmented_frame).data

            frame_pytorch = frame_pytorch[0].cpu().data.transpose(0, 2).transpose(0,1).numpy()[:,:,:]
            frame_pytorch = cv2.cvtColor(frame_pytorch,cv2.COLOR_GRAY2RGB)
            segmented_imagesshow = segmented_frame[0].cpu().data.numpy()[0,:,:]
            frame_pytorch[:,:,2] = frame_pytorch[:,:,2] +  segmented_imagesshow
            if show_reconstruction:
              reconstructed = reconstructed[0].cpu().data.transpose(0, 2).transpose(0,1).numpy()[:,:,:]
              reconstructed = cv2.cvtColor(reconstructed,cv2.COLOR_GRAY2RGB)
              reconstructed[:,:,2] = reconstructed[:,:,2] +  segmented_imagesshow
              cv2.imshow('Reconstructed Frame', reconstructed)

            cv2.imshow('Frame', frame_pytorch)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('p'):
                time.sleep(10000)
            frame_index = frame_index + 1
        except:
            traceback.print_exc()
            return None
    cap.release()
    cv2.destroyAllWindows()



parser = argparse.ArgumentParser(description='Staal Iference from vide.')
parser.add_argument('path', metavar='p', type=str,
                    help='path to a grey-scale video')
parser.add_argument('--reconstruct', metavar='r', type=bool, default = False,
                    help='Show reconstruction')

args = parser.parse_args()

contrast_equalization = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
inference_video(args.path, args.reconstruct)


cv2.waitKey(-1)
