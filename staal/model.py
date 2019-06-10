import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchvision import transforms, utils
import torchvision.models as models
import torch.autograd as autograd

class TextureFinder(nn.Module):

    def __init__(self):
        super(TextureFinder, self).__init__()
        # econder
        # Note:spatial preservation for stride=2: padding = (kernel / 2) - 1
        self.encoder_conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.encoder_conv1.bias.data.zero_()
        # Note: 0.32 is the mean of the images
        self.encoder_conv1.weight.data[:,:,:,:] = (1 / 0.32) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.0001))
        self.encoder_normalization1 = nn.GroupNorm(1, 4, eps=1e-05, affine=True)

        self.encoder_conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.encoder_conv2.bias.data.zero_()
        self.encoder_conv2.weight.data[:,:,:,:] = (1 / (8 * 16))+ torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.0001))
        self.encoder_normalization2 = nn.GroupNorm(1, 16, eps=1e-05, affine=True)

        self.encoder_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.encoder_conv3.bias.data.zero_()
        self.encoder_conv3.weight.data[:,:,:,:] = 1 / (8 * 16)+ torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.001))
        self.encoder_normalization3 = nn.GroupNorm(1, 32, eps=1e-05, affine=True)

        #self.encoder_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True)
        #self.encoder_conv4.bias.data.zero_()
        #self.encoder_conv4.weight.data[:,:,:,:] = 1 / (64 * 16)+ torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.001))

        self.encoder_mu = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.encoder_mu.bias.data.zero_()
        self.encoder_mu.weight.data[:,:,:,:] = (1 / (8 * 16))+ torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.0001))

        self.encoder_log_var = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.encoder_log_var.bias.data[:] = -2.3 # Equivalent to log(0.1)
        self.encoder_log_var.weight.data.zero_()

        # decoder

        self.decoder_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=8, stride=2, padding=3)
        self.decoder_conv1.bias.data.zero_()
        self.decoder_conv1.weight.data[:,:,:,:] = (1 / (8 * 8 * 8))+ torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization1 = nn.GroupNorm(1, 32, eps=1e-05, affine=True)
        
        self.decoder_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=8, stride=2, padding=3, output_padding=0, groups=1, bias=True, dilation=1)
        self.decoder_conv2.bias.data.zero_()
        self.decoder_conv2.weight.data[:,:,:,:] = (1 / (8 * 8 * 8)) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization2 = nn.GroupNorm(1, 64, eps=1e-05, affine=True)

        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2, padding=3, output_padding=0, groups=1, bias=True, dilation=1)
        self.decoder_conv3.bias.data.zero_()
        self.decoder_conv3.weight.data[:,:,:,:] = 1 / (4 * 8 * 8) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.001))
        self.decoder_normalization3 = nn.GroupNorm(1, 32, eps=1e-05, affine=True)

        #self.decoder_conv4 = nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=3, output_padding=0, groups=1, bias=True, dilation=1)
        #self.decoder_conv4.bias.data.zero_()
        #self.decoder_conv4.weight.data[:,:,:,:] = 1 / (32 * 8 * 8) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.001))

        self.decoder_conv5 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=2, padding=3, output_padding=0, groups=1, bias=True, dilation=1)
        # sigmoid can be approximated linearly from (0,1) range using: y(x) = 0.24*x + 0.5
        # 
        self.decoder_conv5.bias.data[:] = - (0.5 / 0.24)
        self.decoder_conv5.weight.data[:,:,:,:] = 1 / (32 * 8 * 8 * 0.24) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.001))

    def forward(self, input):
        # encode
        embeddings_enc0 = F.relu(self.encoder_conv1(input))
        embeddings_enc0 = self.encoder_normalization1(embeddings_enc0)
        embeddings_enc1 = F.relu(self.encoder_conv2(embeddings_enc0))
        embeddings_enc1 = self.encoder_normalization2(embeddings_enc1)
        embeddings_enc2 = F.relu(self.encoder_conv3(embeddings_enc1))
        embeddings_enc2 = self.encoder_normalization3(embeddings_enc2)
        #embeddings_enc3 = F.relu(self.encoder_conv4(embeddings_enc2))
        mu = self.encoder_mu(embeddings_enc2)
        log_var = self.encoder_log_var(embeddings_enc2)
        sample = self.sample_from_mu_log_var(mu, log_var)
        # decode
        embeddings_dec1 = F.relu(self.decoder_conv1(sample, output_size=embeddings_enc2.size()))
        embeddings_dec1 = self.decoder_normalization1(embeddings_dec1)
        embeddings_dec2 = F.relu(self.decoder_conv2(embeddings_dec1, output_size=embeddings_enc1.size()))
        embeddings_dec2 = self.decoder_normalization2(embeddings_dec2)
        embeddings_dec3 = F.relu(self.decoder_conv3(embeddings_dec2, output_size=embeddings_enc0.size()))
        embeddings_dec3 = self.decoder_normalization3(embeddings_dec3)
        #embeddings_dec4 = F.relu(self.decoder_conv4(embeddings_dec3, output_size=embeddings_enc0.size()))
        reconstructed = F.sigmoid(self.decoder_conv5(embeddings_dec3, output_size=input.size()))
        return reconstructed, mu, log_var, sample, embeddings_enc1, embeddings_dec2

    def sample_from_mu_log_var(self, mu, log_var): 
        # sample
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + ( eps * std )
        return sample

class TextureSegmentation(nn.Module):

    def __init__(self):
        super(TextureSegmentation, self).__init__()
       
        self.decoder_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=(8, 16), stride=2, padding=(3, 7))
        self.decoder_conv1.bias.data.zero_()
        self.decoder_conv1.weight.data[:,:,:,:] = (1 / (8 * 8 * 8))+ torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization1 = nn.GroupNorm(1, 32, eps=1e-05, affine=True)
        
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(8, 16), stride=2, padding=(3, 7), output_padding=0, groups=1, bias=True, dilation=1)
        self.decoder_conv2.bias.data.zero_()
        self.decoder_conv2.weight.data[:,:,:,:] = (1 / (8 * 8 * 8)) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization2 = nn.GroupNorm(1, 16, eps=1e-05, affine=True)

        self.decoder_conv3 = nn.ConvTranspose2d(16, 8, kernel_size=(8, 16), stride=2, padding=(3, 7), output_padding=0, groups=1, bias=True, dilation=1)
        self.decoder_conv3.bias.data.zero_()
        self.decoder_conv3.weight.data[:,:,:,:] = 1 / (4 * 8 * 8) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.001))
        self.decoder_normalization3 = nn.GroupNorm(1, 8, eps=1e-05, affine=True)


        self.decoder_conv5 = nn.ConvTranspose2d(8, 1, kernel_size=(8, 16), stride=2, padding=(3, 7), output_padding=0, groups=1, bias=True, dilation=1)
        # sigmoid can be approximated linearly from (0,1) range using: y(x) = 0.24*x + 0.5
        self.decoder_conv5.bias.data[:] = - (0.5 / 0.24)
        self.decoder_conv5.weight.data[:,:,:,:] = 1 / (8 * 8 * 8 * 0.24) + torch.normal(mean = torch.tensor(0.0), std=torch.tensor(0.001))

    def forward(self, sample):
        # decode
        embeddings_dec1 = F.relu(self.decoder_conv1(sample, 
            output_size=torch.empty(sample.size()[0], 32, sample.size()[2] * 2, sample.size()[3] * 2).size()))
        embeddings_dec1 = self.decoder_normalization1(embeddings_dec1)
        embeddings_dec2 = F.relu(self.decoder_conv2(embeddings_dec1, 
            output_size=torch.empty(embeddings_dec1.size()[0], 16, embeddings_dec1.size()[2] * 2, embeddings_dec1.size()[3] * 2).size()))
        embeddings_dec2 = self.decoder_normalization2(embeddings_dec2)
        embeddings_dec3 = F.relu(self.decoder_conv3(embeddings_dec2, 
            output_size=torch.empty(embeddings_dec2.size()[0], 8, embeddings_dec2.size()[2] * 2, embeddings_dec2.size()[3] * 2).size()))
        embeddings_dec3 = self.decoder_normalization3(embeddings_dec3)
        segment = F.sigmoid(self.decoder_conv5(embeddings_dec3, 
            output_size=torch.empty(embeddings_dec3.size()[0], 1, embeddings_dec3.size()[2] * 2, embeddings_dec3.size()[3] * 2).size()))
        return segment


class StyleExtractor(nn.Module):

    def __init__(self):
        super(StyleExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.eval()

    def style(embeddings):
        embeddings = embeddings.view(embeddings.size()[0], embeddings.size()[1], -1)
        embeddings = embeddings - embeddings.mean(2).unsqueeze(2).expand_as(embeddings)
        stds = embeddings.std(2) + 0.01
        embeddings = embeddings / stds.unsqueeze(2).expand_as(embeddings)
        style = torch.bmm(embeddings, embeddings.transpose(1, 2))
        style = style / embeddings.size(2)
        return style

    def forward(self, x):
        x = torch.cat([x,x,x], 1)
        x = (x - 0.456) / 0.225
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        embeddings = self.model.layer1(x)
        embeddings = self.model.layer2(embeddings)
        return StyleExtractor.style(embeddings)