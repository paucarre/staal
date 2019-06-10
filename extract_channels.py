from staal.model import TextureFinder
from staal.image_processing import load_image
import torch
import torch.nn as nn
import cv2


texture_finder = torch.load("model.model").cpu()
texture_finder = torch.load("model.model").cpu()
soldered_3 = load_image("data/images/soldered_3.png")
height = int(soldered_3.size()[1] / 2)
width  = int(soldered_3.size()[2] / 2)
soldered_3 = nn.AdaptiveMaxPool2d((height, width))(soldered_3).data
_, mu_soldered_3, _, samples_soldered_3, _, _ = texture_finder(soldered_3.unsqueeze(0))
print(mu_soldered_3.size())
print(mu_soldered_3.min())
print(mu_soldered_3.max())

image = load_image("data/image-sequences/video2/200.png")
height = int(image.size()[1] / 2)
width  = int(image.size()[2] / 2)
image = nn.AdaptiveMaxPool2d((height, width))(image).data
reconstructed, mus, log_var, samples, fine_embeddings, coarse_embeddings = texture_finder(image.unsqueeze(0))
print(mus.size())
errorc = None
for x_dim in range(mu_soldered_3.size()[2]):
  for y_dim in range(mu_soldered_3.size()[3]):
    current_mu_soldered = mu_soldered_3[0:1, 0:16, x_dim:x_dim+1, y_dim:y_dim+1]
    current_mu_soldered = current_mu_soldered.expand_as(mus)
    print(current_mu_soldered.size())
    error = (current_mu_soldered - mus).abs()
    if errorc is None:
      errorc = error
    else:
      errorc = errorc + error

errorc = errorc / (mu_soldered_3.size()[2] * mu_soldered_3.size()[3])
errshowing = error[0].cpu().data.numpy()[0,:,:]
cv2.imshow("error", errshowing)

imshowing = image.cpu().data.numpy()[0,:,:]
cv2.imshow("input", imshowing)
recoshowing = reconstructed[0].cpu().data.numpy()[0,:,:]
print(imshowing.shape)
print(recoshowing.shape)
cv2.imshow("reconstructed", recoshowing)


for channel in range(samples.size()[1]):
  chshowing = samples[0].cpu().data.numpy()[channel,:,:]
  cv2.imshow(f"{channel}", chshowing)

cv2.waitKey(-1)