from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
import os
import numpy as np
import torch


class TokenisedDataset(Dataset):
    ''' A Dataset wrapper class that returns tokenized input-output pairs from numpy arrays.

    This class wraps a numpy array and splits it into overlapping input-output sequences based on a context window length (cwl).
    It is designed to work with datasets where each element represents a sequence, and the dataset needs to be split into
    overlapping input-output pairs for training.

    Attributes:
        data (np.ndarray): The numpy array containing the data.
        cwl (int): Context window length used to determine the size of input sequences.
    Args:
        NpArr (np.ndarray): A numpy array containing the data to be processed.
        config (object): An object with a 'cwl' attribute specifying the context window length.

    Properties:
        data: The underlying numpy array containing the dataset.
        cwl: The context window length used for splitting data into input-output pairs.

    Methods:
        __len__(): Returns the number of samples in the dataset (total length minus context window).
        __getitem__(idx): Returns a tuple of (input, output) tensors where:
            - input: A tensor containing the sequence from idx to idx + cwl
            - output: A tensor containing the sequence from idx+1 to idx + cwl+1
    '''

    def __init__(self, NpArr, config):
        '''
        Initialises the TokenisedDataset.
        
        Args:
            NpArr (np.ndarray): numpy array with mmap_mode set to read.
            config (object): Configuration object containing 'cwl' attribute for context window length.
        '''
        self.data = NpArr
        self.cwl = config.cwl

    def __len__(self):
        ''' Returns the number of samples in the dataset.

        Returns:
            int: The length of the dataset (total length minus context window).
        '''
        return (len(self.data)-1)//self.cwl

    def __getitem__(self, idx):
        '''Returns a tuple of (input, output) tensors representing a training sample.

        Args:
            idx (int): Index in the dataset.

        Returns:
            tuple: A tuple containing (input_tensor, output_tensor) where:
                - input_tensor: torch tensor of shape (cwl,) from data[idx:idx+cwl]
                - output_tensor: torch tensor of shape (cwl,) from data[idx+1:idx+cwl+1]
        '''
        idx=idx*self.cwl

        x=self.data[idx:idx+self.cwl]
        y=self.data[idx+1:idx+self.cwl+1]
        
        return (
            torch.tensor(x,dtype=torch.long),
            torch.tensor(y,dtype=torch.long)
        )

class DataModule(LightningDataModule):
    ''' Class wrapped around Lightning Data Module to return Train/Val DataLoaders'''
    def __init__(self,file_path,train_val_split,num_workers=2,pin_memory=True,persistent_workers=True,
                 batch_size=32,pre_fetch_factor=2,config=None):
        '''
        Initialises the DataModule.

        Args:
            file_path (str): The path of .npy file
            train_val_split (int): The value split number
            num_workers (int): Number of multiprocessing units for DataLoaders
            pin_memory (bool): Pins memory to a specified RAM section (Increased performance at increased RAM memory usage)
            persistent_workers (bool): Keeps the process active, prevents reintialisation of processes every batch
            batch_size (int): Batch size of DataLoader
            prefetch_factor (int): Number of batches each process fetches.
        '''

        super().__init__()

        self.file_path=file_path
        assert os.path.exists(self.file_path), "File path does not exist"
        self.data=np.load(self.file_path,mmap_mode="r")

        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.persistent_workers=persistent_workers
        self.batch_size=batch_size
        self.pre_fetch_factor=pre_fetch_factor
        self.split=train_val_split
        self.config=config

    def prepare_data(self):
        ''' Checks if data is present '''
        assert self.data is not None , "Dataset Object is not initialised"

    def setup(self, stage=None):
        ''' Setup train and val dataset using predefined class module'''
        split_idx = int(len(self.data) * self.split)
        self.train_dataset=TokenisedDataset(self.data[:split_idx],self.config)
        self.val_dataset=TokenisedDataset(self.data[split_idx:],self.config)

    def train_dataloader(self):
        ''' Returns Train Dataloader'''
        if self.num_workers > 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.pre_fetch_factor,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        ''' Returns Val Dataloader'''
        if self.num_workers > 0:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.pre_fetch_factor,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

