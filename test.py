import torch
import torchvision.utils as utils
from StyleTransfer import StyleTransfer
from PIL import Image
from ImageDataset import ImageDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from ImageDataset import DeNormalize

if __name__ == '__main__':

    model = StyleTransfer()
    model.decoder.load_state_dict(torch.load("models/best_model.pth", map_location='cpu'))

    model.training = False
    model.eval()

    #grab our test images
    #Content images in test_set/content
    #Style Images in test_set/style 
    num_images = 3
    content = ImageDataset(flag='content', root_dir='./test_set/content', data_range=(0,num_images))
    style = ImageDataset(flag='style', root_dir='./test_set/style', data_range=(0,num_images))
    content_img = DataLoader(dataset=content, batch_size=1, shuffle=False)
    style_img = DataLoader(dataset=style, batch_size=1, shuffle=False)

    #since images are normalized before they go through the pretrained vgg19
    #they have to be denormalized with the same stats before they are saved
    denormalizer = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    i = 0
    for content_batch, style_batch in zip(content_img, style_img):

        with torch.no_grad():
            decoded = model.forward(content_batch, style_batch)

        saved = decoded.clone().detach()
        saved = denormalizer(saved)

        utils.save_image(saved, "test_set/results/img" + str(i) + ".png")
        i+=1
