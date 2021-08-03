import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from glob import glob

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for ten, mu, sigma in zip(tensor, self.mean, self.std):
            ten.mul_(sigma).add_(mu)
        return tensor

class ImageDataset(Dataset):
    def __init__(self, flag, root_dir, data_range=(0, 100)):
        
        self.flag = flag
        self.img_names = glob(os.path.join(root_dir, "*.jpg"))[data_range[0]:data_range[1]]
        self.root_dir = root_dir
        print("load " + flag + " dataset start")
        print("    from: %s" % root_dir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        print("Finished loading dataset")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.img_names[idx]).convert("RGB")

        if self.flag == 'train':

            transform = transforms.Compose([
                transforms.Resize(512),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(img)
        else:

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(img)
