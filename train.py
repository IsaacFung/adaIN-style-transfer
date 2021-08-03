import torch
import torch.optim as optim
from ImageDataset import ImageDataset as ImgDataset
from StyleTransfer import StyleTransfer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms, utils
from tqdm import tqdm
import statistics
import torch.nn as nn


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print("Could not find GPU, using CPU instead")

    #hyperparameters
    lr = 1e-4
    num_epochs = 54
    style_weight = 20
    content_weight = 1
    batch_size = 8

    #get our dataset
    #Content set: https://cocodataset.org/#home
    #Style set (wikiart): https://www.kaggle.com/c/painter-by-numbers
    #NOTE: We can't upload all our training data for the submission, so we just grabbed 5 images from each
    data_size = 1200 # change this 5 if you run without adding any images
    train_content_data = ImgDataset(flag='train', root_dir='./train_set/content', data_range=(0,data_size))
    train_style_data = ImgDataset(flag='train', root_dir='./train_set/style', data_range=(0,data_size))
    train_content = DataLoader(dataset=train_content_data, batch_size=batch_size, shuffle=True)
    train_styles = DataLoader(dataset=train_style_data, batch_size=batch_size, shuffle=True)

    #load our model
    model = StyleTransfer().to(device)
    model.training = True

    #we're only training the decoder, so we only need to optimize decoder parameters
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr)
    
    tqdm_tot = min(len(train_content), len(train_content))
    avg_losses = []

    for i in range(num_epochs):
        print('-----------------Epoch = %d-----------------' % (i+1))
        running_loss = []
        for content_batch, style_batch in tqdm(zip(train_content, train_styles), total=tqdm_tot):
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
                        
            decoded, content_loss, style_loss = model.forward(content_batch, style_batch)
            loss = (content_weight * content_loss + style_weight * style_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        
        #average all losses over the epoch so we can plot them later
        avg_loss = statistics.mean(running_loss)
        print("Average Loss = ", avg_loss)
        avg_losses.append(avg_loss)
        
        #save our model every epoch
        torch.save(model.decoder.state_dict(), "models/model_epoch" + str(i) + ".pth")
    
    x = np.arange(len(avg_losses))
    plt.plot(x, avg_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.savefig("loss_plot.png")