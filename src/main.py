import argparse
import numpy as np
import pandas as pd
import pycox
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import models
import losses
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--dataset_path', default='/cluster/tufts/hugheslab/eharve06/MNIST', help='Path to dataset', type=str)
    parser.add_argument('--epochs', default=1000, help='Number of epochs (default: 1000)', type=int)
    parser.add_argument('--experiments_path', default='/cluster/tufts/hugheslab/eharve06/time-to-event/experiments', help='Path to save experiments', type=str)
    parser.add_argument('--hidden_dimension', default=8, help='Hidden dimension (default: 8)', type=int)
    parser.add_argument('--lr', default=0.01, help='Learning rate (default: 0.01)', type=float)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--weight_decay', default=0.01, help='Weight decay (default: 0.01)', type=float)
    args = parser.parse_args()
    
    utils.makedir_if_not_exist(args.experiments_path)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = utils.load_mnist_dataset(args.dataset_path)
    # Suvival MNIST generator returns event_times, censor_times, durations, event_indicators
    _, _, train_durations, train_labels = utils.generate_survival_mnist(y_train, random_state=args.random_state)
    _, _, val_durations, val_labels = utils.generate_survival_mnist(y_val, random_state=args.random_state)
    _, _, test_durations, test_labels = utils.generate_survival_mnist(y_test, random_state=args.random_state)
    
    mean = np.mean(X_train)
    std = np.std(X_train)
    normalize = lambda image: (image-mean)/std
    
    train_dataset = utils.Dataset(X_train, train_durations, train_labels, normalize)
    val_dataset = utils.Dataset(X_val, val_durations, val_labels, normalize)
    test_dataset = utils.Dataset(X_test, test_durations, test_labels, normalize)
    
    shuffled_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, collate_fn=utils.collate_fn)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.Net(hidden_dimension=args.hidden_dimension).to(device)
    criterion = losses.MSEHinge()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    columns = ['epoch', 'train_CI', 'train_loss', 'test_CI', 'test_loss', 'val_CI', 'val_loss']
    model_history_df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):
        
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)
        val_metrics = utils.evaluate(model, criterion, val_loader)
        test_metrics = utils.evaluate(model, criterion, test_loader)
        
        # Append evaluation metrics to DataFrame
        row = [epoch, train_metrics['CI'], train_metrics['loss'], test_metrics['CI'], test_metrics['loss'], val_metrics['CI'], val_metrics['loss']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_path}/{args.model_name}.csv', index=False)
        
    torch.save({
        'train_durations': train_metrics['durations'],
        'train_labels': train_metrics['labels'],
        'train_logits': train_metrics['logits'],
        'test_durations': test_metrics['durations'],
        'test_labels': test_metrics['labels'],
        'test_logits': test_metrics['logits'],
        'val_durations': val_metrics['durations'],
        'val_labels': val_metrics['labels'],
        'val_logits': val_metrics['logits'],
    }, f'{args.experiments_path}/{args.model_name}.pt')
