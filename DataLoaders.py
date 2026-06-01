from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
import os
import numpy as np

class TokenisedDataset(Dataset):
    ''' A Dataset wrapped class module to return tokenized input-output pairs'''

    def __init__(self,NpArr,config):
        '''
        Initialising the Dataset.
        Args:
            NpArr (arr): numpy array with mmap_mode set to read.
        '''
        self.data=NpArr
        self.cwl=config.cwl
        
    def __len__(self):
        ''' Returns the length of dataset'''
        return len(self.data)
    
    def __getitem__(self,idx):
        '''Returns the input-output pair'''
        x=self.data[idx:idx+self.cwl]
        y=self.data[idx+1:idx+self.cwl+1]
        return x,y
    

class DataModule(LightningDataModule):
    ''' Class wrapped around Lightning Data Module to return Train/Val DataLoaders'''
    def __init__(self,file_path,train_val_split,num_workers=2,pin_memory=True,persistent_workers=True,
                 batch_size=32,prefetch_factor=2,config=None):
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
        self.persistant_workers=persistent_workers
        self.batch_size=batch_size
        self.prefetch_factor=prefetch_factor
        self.split=train_val_split
        self.config=config

    def prepare_data(self):
        ''' Checks if data is present '''
        assert self.data is not None , "Dataset Object is not initialised"

    def setup(self):
        ''' Setup train and val dataset using predefined class module'''
        split_val=len(self.data)*self.split
        self.train_dataset=TokenisedDataset(self.data[:split_val],self.config)
        self.val_dataset=TokenisedDataset(self.data[split_val:],self.config)

    def train_dataloader(self):
        ''' Returns Train Dataloader'''
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistant_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        ''' Returns Val Dataloader'''
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistant_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
    
