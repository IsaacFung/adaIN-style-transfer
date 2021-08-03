import torch

def average(x):
    N, C, H, W = x.shape
    # mean over HxW dimensions and using view to expand dimension so it matches tensor
    feature_mean = torch.mean(x, dim=[2,3]).view(N, C, 1, 1)
    return feature_mean

def stdev(x):
    N, C, H, W = x.shape
    # 1e-10 is a small value to prevent division by 0
    feature_std = torch.std(x, dim=[2,3]).view(N, C, 1, 1) + 1e-10
    return feature_std

if __name__ == "__main__":
    #for testing purposes
    test = torch.arange(9,  dtype=torch.float64)  
    t = test.view(1,1,3,3)
    print(t)
    print(average(t))
    print(average(t).shape)
    print(stdev(t))