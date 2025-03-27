# ============================================================================
# File: p2_train.py
# Date: 2025-03-11
# Author: TA
# Description: Training a model and save the best model.
# ============================================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

import config as cfg
from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

def plot_learning_curve(
        logfile_dir: str,
        result_lists: dict
    ):
    '''
    Plot and save the learning curves under logfile_dir.
    - Args:
         - logfile_dir: str, the directory to save the learning curves.
         - result_lists: dict, the dictionary contains the training and
                         validation results with keys
                         'train_acc', 'train_loss', 'val_acc', 'val_loss'.
     - Returns:
         - None
    '''
    # Create the directory if it doesn't exist
    os.makedirs(logfile_dir, exist_ok=True)

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(result_lists['train_acc'], label='Train Accuracy')
    plt.plot(result_lists['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(logfile_dir, 'accuracy_curve.png'))
    plt.close()

    # Plot training and validation loss
    plt.figure()
    plt.plot(result_lists['train_loss'], label='Train Loss')
    plt.plot(result_lists['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(logfile_dir, 'loss_curve.png'))
    plt.close()

def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        unlabeled_loader: torch.utils.data.DataLoader,  # Add unlabeled dataloader
        logfile_dir: str,
        model_save_dir: str,
        criterion: nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim,
        device: torch.device
    ):
    '''
    Training and validation process with semi-supervised learning.
    '''
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()

        # Process labeled data
        for batch, data in enumerate(train_loader):
            images, labels = data['images'].to(device), data['labels'].to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()

        # Process unlabeled data
        for batch, data in enumerate(unlabeled_loader):
            images = data['images'].to(device)
            pred = model(images)
            pseudo_labels = torch.argmax(pred, dim=1)  # Generate pseudo-labels
            loss = criterion(pred, pseudo_labels)  # Use pseudo-labels for loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate training metrics
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc.cpu().numpy())
        train_loss_list.append(train_loss)
        print(f'[{epoch + 1}/{cfg.epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')

        ##### VALIDATION #####
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            for batch, data in enumerate(val_loader):
                sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Val batch: {batch + 1} / {len(val_loader)}')
                sys.stdout.flush()
                # Data loading
                images, labels = data['images'].to(device), data['labels'].to(device)
                # Forward pass
                pred = model(images)
                # Calculate loss
                loss = criterion(pred, labels)
                # Evaluate
                val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                val_loss += loss.item()

        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc.cpu().numpy())
        val_loss_list.append(val_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        
        # Scheduler step
        scheduler.step()

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir, 'result_log.txt'),
                         epoch, epoch_time,
                         train_acc, val_acc,
                         train_loss, val_loss,
                         is_better)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f'[{epoch + 1}/{cfg.epochs}] Save best model to {model_save_dir} ...')
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, 'model_best.pth'))
            best_acc = val_acc

        ##### PLOT LEARNING CURVE #####
        ##### TODO: check plot_learning_curve() in this file #####
        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }
        plot_learning_curve(logfile_dir, current_result_lists)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', 
                        help='dataset directory', 
                        type=str, 
                        default='../hw2_data/p2_data/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Experiment name
    exp_name = cfg.model_type \
        + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') \
        + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    # Fix a random seed for reproducibility
    set_seed(2025)

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    if cfg.model_type == 'mynet':
        model = MyNet()
    elif cfg.model_type == 'resnet18':
        model = ResNet18()
    else:
        raise NameError('Unknown model type')

    model.to(device)

    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    train_loader = get_dataloader(os.path.join(dataset_dir, 'train'),
                                  batch_size=cfg.batch_size, split='train')
    val_loader   = get_dataloader(os.path.join(dataset_dir, 'val'),
                                  batch_size=cfg.batch_size, split='val')
    unlabeled_loader = get_dataloader(
        os.path.join(dataset_dir, 'unlabel'),
        batch_size=cfg.batch_size,
        unlabeled=True  # Use UnlabeledDataset
    )
    # Load unlabeled data
    unlabeled_loader = get_dataloader(os.path.join(dataset_dir, 'unlabel'),
                                      batch_size=cfg.batch_size, unlabeled=True)

    ##### LOSS & OPTIMIZER #####
    criterion = nn.CrossEntropyLoss()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                    momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=0.1)
    
    ##### TRAINING & VALIDATION #####
    ##### TODO: check train() in this file #####
    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          unlabeled_loader=unlabeled_loader,  # Pass unlabeled dataloader
          logfile_dir=logfile_dir,
          model_save_dir=model_save_dir,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device)
    
if __name__ == '__main__':
    main()
