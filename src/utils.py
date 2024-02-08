import os
import sys
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
# PyTorch
import torch
import torchvision
import torchmetrics

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_mnist_dataset(dataset_path):
    
    if not os.path.isfile(f'{dataset_path}/mnist.pkl.gz'):
        from urllib.request import urlretrieve
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print(f'Downloading data from {origin}')
        makedir_if_not_exist(dataset_path)
        urlretrieve(origin, dataset_path)

    f = gzip.open(f'{dataset_path}/mnist.pkl.gz', 'rb')
    if sys.version_info[0] == 3:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
    else:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    return train_set, valid_set, test_set

def generate_survival_mnist(labels, random_state=42):
    
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')
    
    mean_survival_times = {0: 11.25, 1: 2.25, 2: 5.25, 3: 5.0, 4: 4.75, 5: 8.0, 6: 2.0, 7: 11.0, 8: 1.75, 9: 10.75}
    mean = np.array([mean_survival_times[label] for label in labels])
    variance = 10e-3
    scale = variance / mean
    shape = mean / scale
    event_times = random_state.gamma(shape, scale=scale)
    low, high = np.min(event_times), np.percentile(event_times, 90)
    censor_times = random_state.uniform(low=low, high=high, size=event_times.shape)
    durations = np.min([event_times, censor_times], axis=0)
    event_indicators = (event_times<=censor_times).astype(int)
    return event_times, censor_times, durations, event_indicators

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}

    for images, labels in dataloader:
                        
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        probabilities = torch.softmax(logits, dim=1)
        acc1, acc5 = accuracy(probabilities, labels, topk=(1, 5))
        metrics['acc1'] += batch_size/dataset_size*acc1.item()
        metrics['acc5'] += batch_size/dataset_size*acc5.item()
        metrics['loss'] += batch_size/dataset_size*loss.item()
        
        if lr_scheduler:
            lr_scheduler.step()
            
    return metrics

def evaluate(model, criterion, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()   
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}
            
    with torch.no_grad():
        for images, labels in dataloader:
                        
            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)
                
            logits = model(images)
            loss = criterion(logits, labels)
            
            batch_size = len(images)
            probabilities = torch.softmax(logits, dim=1)
            acc1, acc5 = accuracy(probabilities, labels, topk=(1, 5))
            metrics['acc1'] += batch_size/dataset_size*acc1.item()
            metrics['acc5'] += batch_size/dataset_size*acc5.item()
            metrics['loss'] += batch_size/dataset_size*loss.item()
    
    return metrics