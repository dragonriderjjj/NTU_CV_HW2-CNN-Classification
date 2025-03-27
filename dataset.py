# ============================================================================
# File: dataset.py
# Date: 2025-03-11
# Author: TA
# Description: Dataset and DataLoader.
# ============================================================================

import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloader(
        dataset_dir,
        batch_size: int = 1,
        split: str = 'test',
        unlabeled: bool = False):  # Add `unlabeled` flag
    '''
    Build a dataloader for given dataset and batch size.
    - Args:
        - dataset_dir: str, path to the dataset directory
        - batch_size: int, batch size for dataloader
        - split: str, 'train', 'val', or 'test'
        - unlabeled: bool, whether to load unlabeled data
    - Returns:
        - dataloader: torch.utils.data.DataLoader
    '''
    if unlabeled:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = UnlabeledDataset(dataset_dir, transform=transform)
    else:
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:  # 'val' or 'test'
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        dataset = CIFAR10Dataset(dataset_dir, split=split, transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(split == 'train'),
                            num_workers=0,
                            pin_memory=True,
                            drop_last=(split == 'train'))

    return dataloader

class UnlabeledDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        super(UnlabeledDataset).__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_names = os.listdir(self.dataset_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.image_names[index])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'images': image}
    
    
class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None):
        super(CIFAR10Dataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)
        
        self.image_names = json_data['filenames']
        if self.split != 'test':
            self.labels = json_data['labels']

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        '''
        Get an item from the dataset.
        - Args:
            - index: int, the index of the item to retrieve
        - Returns:
            - A dictionary containing 'images' and 'labels' (if not test set)
        '''
        # Load the image
        image_path = os.path.join(self.dataset_dir, self.image_names[index])
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format

        # Apply the transform
        if self.transform:
            image = self.transform(image)

        # If not the test set, return image and label
        if self.split != 'test':
            label = torch.tensor(self.labels[index], dtype=torch.long)
            return {
                'images': image,
                'labels': label
            }
        else:
            # For the test set, return only the image
            return {
                'images': image
            }