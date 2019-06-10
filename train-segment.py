from staal.model import TextureSegmentation
from staal.model import StyleExtractor
from staal.image_processing import load_image
from staal.dataset import SegmentationDataset
from staal.dataset import SegmentationSequenceDataset
from staal.loss import binary_cross_entropy
from staal.loss import style_similarity
from staal.loss import kl_divergence
import torch
import torch.nn as nn
import cv2


best_loss = 1
epochs=10000
batch_size = 4
learning_rate = 0.01
texture_segmentation = torch.load("models/texture-segment.model")#TextureSegmentation().cuda()#
texture_finder = torch.load("models/model.model")
optimizer = torch.optim.Adam(texture_segmentation.parameters(), lr = learning_rate)
for epoch in range(epochs):
    dataset_train = SegmentationDataset()
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)
    print("Supervised")
    show_sample = True
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()
        _, _, _, samples, _, _ = texture_finder(images)
        segmented_images = texture_segmentation(samples)
        optimizer.zero_grad()
        binary_cross_entropy_error = binary_cross_entropy(segmented_images, target)
        binary_cross_entropy_error.backward()
        optimizer.step()
        current_loss = binary_cross_entropy_error.data
        print(f"Current Supervised loss: {current_loss}")
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"New Supervised best loss: {best_loss}")
            torch.save(texture_segmentation, "models/texture-segment.model")
        imshowing = images[0].cpu().data.numpy()[0,:,:]
        if cv2.waitKey(1) & 0xFF == ord('s'):
            show_sample = True
        if show_sample:
            cv2.imshow("inputs", imshowing)
            segmented_imagesshow = segmented_images[0].cpu().data.numpy()[0,:,:]
            cv2.imshow("segmented_images", segmented_imagesshow)
            targetshow = target[0].cpu().data.numpy()[0,:,:]
            cv2.imshow("target", targetshow)
            show_sample = False
'''

    dataset_train = SegmentationSequenceDataset()
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)
    print("Un Supervised")
    for i, (images, images_next) in enumerate(train_loader):
        images = images.cuda()
        images_next = images_next.cuda()
        _, _, _, samples, _, _ = texture_finder(images)
        _, _, _, samples_next, _, _ = texture_finder(images_next)
        segmented_images = texture_segmentation(samples).mean()
        segmented_images_next = texture_segmentation(samples_next).mean().data
        optimizer.zero_grad()
        error = (segmented_images - segmented_images_next).abs().view(1) *0.01
        error.backward()
        optimizer.step()
        current_loss = error.data
        print(f"Current Sequence loss: {current_loss}")
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"New best Sequence loss: {best_loss}")
            torch.save(texture_segmentation, "texture-segment.model")
'''
