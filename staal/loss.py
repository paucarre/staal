import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_cross_entropy(reconstructed, original_input):
    #reconstructed = nn.AdaptiveMaxPool2d((8, 8))(reconstructed)
    #original_input = nn.AdaptiveMaxPool2d((8, 8))(original_input)
    return F.binary_cross_entropy(reconstructed.view(reconstructed.size()[0], -1), 
       original_input.view(original_input.size()[0], -1), reduction="mean")

def kl_divergence( mu, log_var):
    return -0.5 * torch.mean(1 + log_var.view(log_var.size()[0], -1) - mu.view(mu.size()[0], -1).pow(2) - log_var.view(log_var.size()[0], -1).exp())

def style_similarity(original, regenerated, style_extractor):
    style_sample = style_extractor(original)
    style_regenerated = style_extractor(regenerated)
    return (style_regenerated - style_sample).abs().mean()

