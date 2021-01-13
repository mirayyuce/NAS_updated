from __future__ import absolute_import
import numpy as np
import random
import torch
import copy
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision


def prepare_data(batch_size=128, valid_frac=0.3):
    """
    Data preprocessing method. Data cleaning, augmentation, validation set split
    
    Args:
        batch_size: minibatch size
        valid_frac: validation data fraction

    Returns:
        Returns data loader objects
    """
    # Data augmentation
    n_holes = 1
    length = 16

    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0,translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_transform.transforms.append(Cutout(n_holes=n_holes, length=length))
    
    # Define dataset objects
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=train_transform, download=True)
    valid_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=test_transform, download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_frac * num_train))
    manual_seed=10
    np.random.seed(manual_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Create datasets, iterable on batches
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader, test_loader


def mixup_data(x, y, alpha=1.0):
    """
    Compute the mixup data.

    mixup trains a neural network on convex combinations of pairs of examples and their labels.
    By doing so, mixup regularizes the neural network to favor simple linear behavior
    in-between training examples. https://arxiv.org/pdf/1710.09412.pdf
    
    Args:
        x: data
        y: labels
        alpha: optimizer alpha
        use_cuda: flag to use 

    Returns:
        Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()
    

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Randomly mask out one or more patches from an image.

        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def GetSubsequentLayers(layer_id, model_descriptor):
    """
    Method to find the subsequent layers of a given layer
    
    Args:
        layer_id: layer 
        model_descriptor: model_descriptor, look at crate_vanilla =_network.py

    Returns:
        Returns a list of subsequent layers, if there is one single subsequent layer, and whether it is conv layer
    """
    subsequentlayers = [layer for layer in model_descriptor['layers'] if layer_id in layer['input']]

    if len(subsequentlayers) == 1:
        SingleSSL = 1
    else:
        SingleSSL = 0

    if SingleSSL == 1 and subsequentlayers[0]['type'] == 'conv':
        IsConv = 1
    else:
        IsConv = 0

    return [subsequentlayers, SingleSSL, IsConv]


def ReplaceInput(layers, layer_id2remove, layer_id2add):
    """
    Method to replace one layer's input with the other's
    
    Args:
        layers: model layers
        layer_id2remove: old layer id
        layer_id2add: new layer id

    Returns:
    """

    for layer in layers:
        layer['input'] = [int(layer_id2add) if x == int(layer_id2remove) else x for x in layer['input']]


def GetUnusedID(model_descriptor):
    """
    Method to find an unused layer id
    
    Args:
        model_descriptor: model_descriptor

    Returns:
        Unused id
    """

    unused_id = np.max([layers['id'] for layers in model_descriptor['layers']]) + 1

    return unused_id


def InheritWeights(old_model, new_model):
    """
    Method to load old models weigths to the new model
    
    Args:
        old_model: old model
        new_model: new model
    Returns:
        new model
    """
    weightspath = 'tempweights/' + str(random.randint(1, 10 ** 12))

    torch.save(old_model.state_dict(), weightspath)

    new_model.load_state_dict(torch.load(weightspath), strict=False)

    return new_model
