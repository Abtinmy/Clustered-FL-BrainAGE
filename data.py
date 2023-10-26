import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OpenBHBDataset(Dataset):
    def __init__(self, data_dir, tsv_path, modality, transform=None):
        self.data_dir = data_dir
        self.tsv_file = pd.read_csv(tsv_path, sep='\t')
        self.transform = transform
        self.modality = modality

    def __len__(self):
        return len(self.tsv_file)

    def __getitem__(self, index):
        id = self.tsv_file.iloc[index]['participant_id']
        image = np.load(f'{self.data_dir}/sub-{id}_preproc-{self.modality}_T1w.npy').astype('float64')
        image = image / np.max(image)

        if self.transform:
            image = self.transform(np.squeeze(image))

        age = self.tsv_file.iloc[index]['age']
        y_label = torch.tensor(float(age))
        return image, y_label


def generate_dataloaders(args):
    train_dataloaders, test_dataloaders = [], []
    server_dataloader = None

    for client in range(1, args.num_clients + 1):
        train_dataset = OpenBHBDataset(args.data_dir + 'train_quasiraw',
                                       f'{args.data_dir}/{args.name_experiment}/{client}_train.tsv',
                                       modality='quasiraw',
                                       transform=transforms.ToTensor())
        train_dataloaders.append(DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size))

        test_dataset = OpenBHBDataset(args.data_dir + 'val_quasiraw',
                                      f'{args.data_dir}/{args.name_experiment}/{client}_test.tsv',
                                      modality='quasiraw',
                                      transform=transforms.ToTensor())
        test_dataloaders.append(DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size))

        server_dataset = OpenBHBDataset(args.data_dir + 'train_quasiraw',
                                        f'{args.data_dir}/{args.name_experiment}/server.tsv',
                                        modality='quasiraw',
                                        transform=transforms.ToTensor())
        server_dataloader = DataLoader(server_dataset, shuffle=False, batch_size=args.batch_size)

    return train_dataloaders, test_dataloaders, server_dataloader
