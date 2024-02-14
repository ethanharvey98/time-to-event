import os
import sys
import pickle
import gzip
import numpy as np
from lifelines.utils import concordance_index
# PyTorch
import torch
import torchvision
    
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, durations, labels, transform=None):
        self.images = images
        self.durations = durations
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.images[index], self.durations[index], self.labels[index]) if self.transform == None else (self.transform(self.images[index]), self.durations[index], self.labels[index])
    
def collate_fn(batch):
    images, durations, labels = zip(*batch)
    images = np.concatenate(images, axis=0)
    durations = np.array(durations)
    labels = np.array(labels)
    return torch.tensor(images).view(-1, 1, 28, 28), torch.tensor(durations), torch.tensor(labels)

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'CI': 0.0, 'durations': np.array([]), 'labels': np.array([]), 'logits': np.array([]), 'loss': 0.0}

    for images, durations, labels in dataloader:
                        
        if device.type == 'cuda':
            images, durations, labels = images.to(device), durations.to(device), labels.to(device)

        model.zero_grad()
        logits = model(images)
        loss = criterion(logits, durations, labels)
        loss.backward()
        optimizer.step()
        
        batch_size = len(images)
        metrics['loss'] += batch_size/dataset_size*loss.item()
        
        if device.type == 'cuda':
            durations, labels, logits = durations.cpu(), labels.cpu(), logits.cpu()
        
        metrics['durations'] = np.append(metrics['durations'], durations.numpy())
        metrics['labels'] = np.append(metrics['labels'], labels.numpy())
        metrics['logits'] = np.append(metrics['logits'], logits.detach().numpy())
        
        if lr_scheduler:
            lr_scheduler.step()
            
    metrics['CI'] = concordance_index(metrics['durations'], metrics['logits'], metrics['labels'])
            
    return metrics

def evaluate(model, criterion, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()   
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'CI': 0.0, 'durations': np.array([]), 'labels': np.array([]), 'logits': np.array([]), 'loss': 0.0}
            
    with torch.no_grad():
        for images, durations, labels in dataloader:

            if device.type == 'cuda':
                images, durations, labels = images.to(device), durations.to(device), labels.to(device)
                
            logits = model(images)
            loss = criterion(logits, durations, labels)
            
            batch_size = len(images)
            metrics['loss'] += batch_size/dataset_size*loss.item()
    
            if device.type == 'cuda':
                durations, labels, logits = durations.cpu(), labels.cpu(), logits.cpu()

            metrics['durations'] = np.append(metrics['durations'], durations.numpy())
            metrics['labels'] = np.append(metrics['labels'], labels.numpy())
            metrics['logits'] = np.append(metrics['logits'], logits.detach().numpy())
                    
        metrics['CI'] = concordance_index(metrics['durations'], metrics['logits'], metrics['labels'])
    
    return metrics