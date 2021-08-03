# -*- coding: utf-8 -*-
"""NeuralstyleTransfer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yq4apfay9QPTeXQc5986htgLzRBuhDE-
"""

"""
This is the optimization based approach used by Gatys et al.
We used this to have images to compare our AdaIN transfer to.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import MSELoss
from torchvision import transforms, utils
from torchvision.models import vgg
from PIL import Image


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Could not find GPU, using CPU instead")

transformation = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def preprocess_image(img):
    output = transformation(img)
    return output.unsqueeze(0)

loss = MSELoss()
def content_loss(input, target):
    input.squeeze()
    target.squeeze()
    # print(input)
    # print(target)
    # los = loss(input, target)
    # print(los)
    return loss(input, target)

def gram_matrix(x):
    N, C, H, W = x.shape
    features = x.view(N*C, H*W)
    gram_matrix = torch.mm(features, features.t())
    normalized_gram_matrix = gram_matrix/(N*C*H*W)
    return normalized_gram_matrix

def style_loss(input_activation, target_activation):
    total = 0
    for map1, map2 in zip(input_activation, target_activation):
        gram1 = gram_matrix(map1)
        gram2 = gram_matrix(map2)
        total = total + 1/5*loss(gram1, gram2)
        
        

    return total

class Pretrained_VGG(nn.Module):
    def __init__(self, layers_name):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features[:29].eval().to(device)
        self.layer_index = {'conv1_1':0, 'conv2_1':5, 'conv3_1':10, 'conv4_1':19, 'conv4_2':21, 'conv5_1':28}
        self.selected = layers_name
        self.activations = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.clone() 
        return hook 

    def forward(self, x):
        for name in self.selected:
            #print(name)
            self.vgg[self.layer_index[name]].register_forward_hook(self.get_activation(name))
        
        output = self.vgg(x)
        return output, self.activations

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for ten, mu, sigma in zip(tensor, self.mean, self.std):
            ten.mul_(sigma).add_(mu)
        return tensor

# Hyperparameters
num_epoch = 300
alpha = 1e5
beta = 1

content_names = ['conv4_2']
style_names = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']

# Import images and preprocess
content = Image.open("floodedhouses.jpg")
style = Image.open("starynight.jpg").resize(content.size)
white = Image.new('RGB', (content.size), (255, 255, 255))

content_processed = preprocess_image(content).to(device)
style_processed = preprocess_image(style).to(device)
print(content_processed.size())
# trained_imaged = preprocess_image(white).to(device).requires_grad_(True)
trained_imaged = preprocess_image(content).to(device).requires_grad_(True)


# Initializing the utility objects
denormalizer = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# optimizer = optim.Adam([trained_imaged], lr=learning_rate)
# optimizer = optim.Adadelta([trained_imaged], lr=learning_rate)
optimizer = optim.LBFGS([trained_imaged])

model_content = Pretrained_VGG(list(content_names))
model_style = Pretrained_VGG(style_names)
model_train = Pretrained_VGG(style_names+content_names)

losses = []

for i in range(num_epoch):

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        print(f"Starting Epoch {i+1}")
        vgg_content, content_act = model_content(content_processed)
        vgg_style, style_act = model_style(style_processed)
        vgg_trained_imaged, act = model_train(trained_imaged)

        #print(style_act==act)
        contentloss = content_loss( content_act[content_names[0]] , act[content_names[0]])
        styleloss = style_loss(list(style_act.values()), list(act.values()))
        totalloss = beta * contentloss + alpha * styleloss
        
        print("content loss:", contentloss)
        print("style loss:", styleloss)
        print("total loss:", totalloss)
        losses.append(totalloss.item())

        if totalloss.requires_grad:
            totalloss.backward()
        return totalloss

    if i%5==0:
        saved = trained_imaged.clone().detach()
        saved = denormalizer(saved)
        utils.save_image(saved, f"{i}.jpg")

    optimizer.step(closure)
    # if contentloss <40 and styleloss<0.005:
    #     break


x = np.arange(len(losses))
plt.plot(x, losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
#plt.savefig("loss_plot.png")


saved = trained_imaged.clone().detach()
saved = denormalizer(saved)
utils.save_image(saved, "notpoop.jpg")