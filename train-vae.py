from staal.model import TextureFinder
from staal.model import StyleExtractor
from staal.image_processing import load_image
from staal.dataset import ImageDataset
from staal.loss import binary_cross_entropy
from staal.loss import style_similarity
from staal.loss import kl_divergence
import torch
import torch.nn as nn
import cv2


best_loss = 1
epochs=10000
batch_size = 64
learning_rate = 0.01
style_extractor = StyleExtractor().cuda()
texture_finder = torch.load("models/model.model")#TextureFinder().cuda() 
dataset_train = ImageDataset()
train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=1)
optimizer = torch.optim.Adam(texture_finder.parameters(), lr = learning_rate)
for epoch in range(epochs):
    for i, images in enumerate(train_loader):
        images = images.cuda()
        target = images.clone()
        target.requires_grad = False
        reconstructed, mus, log_var, samples, fine_embeddings, coarse_embeddings = texture_finder(images)
        optimizer.zero_grad()
        kl_divergence_lower_bound = kl_divergence( mus, log_var) * 0.001
        binary_cross_entropy_error = binary_cross_entropy(reconstructed, target) * 0.01
        style_similarity_error = style_similarity(images, reconstructed, style_extractor)
        total_loss = style_similarity_error + kl_divergence_lower_bound + binary_cross_entropy_error
        total_loss.backward()
        optimizer.step()
        current_loss = total_loss.data
        print(f"Current loss: {current_loss} | Style Loss {style_similarity_error} | " \
                f"BCE Loss {binary_cross_entropy_error} | KL {kl_divergence_lower_bound}")
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"New best loss: {best_loss} | Style Loss {style_similarity_error} | " \
                f"BCE Loss {binary_cross_entropy_error} | KL {kl_divergence_lower_bound }")
            torch.save(texture_finder, "models/model.model")
        if i % 5 == 0:
                imshowing = images[0].cpu().data.numpy()[0,:,:]
                cv2.imshow("inputs", imshowing)
                recshow = reconstructed[0].cpu().data.numpy()[0,:,:]
                cv2.imshow("reconstructed", recshow)
                samplesshow = samples[0].cpu().data.numpy()[0,:,:]
                cv2.imshow("samples", samplesshow)