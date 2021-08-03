import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from util import average
from util import stdev
from copy import deepcopy
from AdaIN import AdaIN


#grabs the activation layers
activations = [None] * 4
def style_hook(idx, module, input, output):
	activations[idx] = output.clone()

def ContentLoss(dec_features, adain_out):
	loss = nn.MSELoss()
	return loss(dec_features, adain_out)

def StyleLoss(dec_activations, style_activations):
    # NxCxHxW
	loss = nn.MSELoss()
	mean_sum = 0
	std_sum = 0
	for dec_activation, style_activation in zip(dec_activations, style_activations):

		dec_avg = average(dec_activation)
		dec_std = stdev(dec_activation)
		style_avg = average(style_activation)
		style_std = stdev(style_activation)

		mean_sum = mean_sum + loss(dec_avg, style_avg)
		std_sum = std_sum + loss(dec_std, style_std)
	
	return mean_sum + std_sum


class StyleTransfer(nn.Module):

	def __init__(self):
		super().__init__()

		self.vgg19 = models.vgg19(pretrained=True).features.eval()
		# relu1_1, relu2_1, relu3_1, relu4_1 and their index
		self.style_layers = [1, 6, 1, 20]
		i = 0
		for l in self.style_layers:
			self.vgg19[l].register_forward_hook(partial(style_hook, i))
			i = i + 1
		self.vgg19 = self.vgg19[:21]
			
		self.adaIN = AdaIN()
		self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )


	def forward(self, content, style):
		
		self.content_features = self.vgg19(content)
		self.style_features = self.vgg19(style)
		self.adain_out = self.adaIN(self.content_features, self.style_features)
		self.decoded = self.decoder(self.adain_out)
		
		if self.training:
			content_loss, style_loss = None, None
			
			#save a copy of the activations from the style image
			style_activations = activations.copy()

			#run decoded through encoder and save activations
			dec_output = self.vgg19(self.decoded)
			dec_activations = activations.copy()

			#compute losses
			content_loss = ContentLoss(dec_output, self.adain_out)
			style_loss = StyleLoss(dec_activations, style_activations)

			return self.decoded, content_loss, style_loss

		else:
			return self.decoded
		
	
