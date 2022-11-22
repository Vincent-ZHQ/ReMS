import os
import logging
import pickle as pkl
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

__all__ = ['MMDataLoader']

logger = logging.getLogger('MER')

class MMDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode + '.pkl'
        self.save = []

        self.__init_dataset()

    def __init_dataset(self):
        with open(os.path.join(self.args.data_dir,  self.mode), 'rb') as f:
            data = pkl.load(f)

        self.text = np.array(data['text_bert']).astype(np.float32)
        self.video = np.array(data['video']).astype(np.float32)
        self.audio = np.array(data['audio_mfcc']).astype(np.float32)
        self.labels = np.array(data['labels']).astype(np.int_)

        self.audio[self.audio == -np.inf] = 0

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        text = torch.tensor(self.text[index])
        audio = torch.tensor(self.audio[index])
        video = torch.tensor(self.video[index])
        labels = torch.tensor(self.labels[index])
        return audio, video, text, labels

def MMDataLoader(args):

    train_set = MMDataset(args, 'train')
    valid_set = MMDataset(args, 'valid')
    test_set = MMDataset(args, 'test')

    print("Train Dataset: ", len(train_set))
    print("Valid Dataset: ", len(valid_set))
    print("Test Dataset: ", len(test_set))

    # print(args.num_workers, args.batch_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=False, drop_last=True)
    valid_loader = DataLoader(valid_set,  batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True)

    return train_loader, valid_loader, test_loader


