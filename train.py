from Model_Classes import GPT
from Optimizer import Hybrid_Optim_with_Cosine_Scheduler,HybridOptim
from HyperParam_Classes import GPTConfig
from HyperParam_Classes import TrainParams,OptimHParams
import lightning.pytorch as pl
from torchinfo import summary
import torch.nn as nn
from DataLoaders import DataModule


class Train_Model(pl.LightningModule):
    '''Class wrapped around Lightning Module to train the GPT model using the Hybrid Optimizer with Cosine Scheduler.'''
    def __init__(self,config,):
        '''
        Initialises the class.
        Args:
            config (Config DataClass Object): The Config dataclass object containing the hyperparameters of the model.'''
        super().__init__()
        self.save_hyperparameters()

        self.model=GPT(config)
        self.model.compile()
        self.loss_fn=nn.CrossEntropyLoss()
        self.config=config

    def forward(self,x):
        ''' Calls the forward function of the GPT model.
        Args:
            x (Tensor): Input Tensor of shape (B,T) where B is the batch size and T is the context window length.
        Returns:
            Tensor of shape (B,T,C) where C is the vocab size.
        '''
        return self.model(x)
    
    def training_step(self,batch):
        ''' Calls the forward function and calculates the loss for a batch of data.
        Args:            
            batch (Tensor): A batch of input-output pairs of shape (B,T) where B is the batch size and T is the context window length.
        Returns:
            loss (Tensor): The calculated loss for the batch.
        '''
        x,y=batch
        logits=self(x)

        loss=self.loss_fn(logits.view(-1,logits.size(-1)),y.view(-1))
        self.log("train_loss",loss,prog_bar=True)
        return loss
    
    def validation_step(self,batch):
        ''' Calls the forward function and calculates the loss for a batch of validation data.
        Args:            
            batch (Tensor): A batch of input-output pairs of shape (B,T) where B is the batch size and T is the context window length.
        Returns:
            loss (Tensor): The calculated loss for the batch.
        '''
        x,y=batch
        logits=self(x)

        loss=self.loss_fn(logits.view(-1,logits.size(-1)),y.view(-1))
        self.log("val_loss",loss,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        ''' Configures the optimizers for training using the Hybrid Optimizer with Cosine Scheduler.
        Returns:
            optimizer (Optimizer): The configured optimizer for training.
        '''
        optimizer=Hybrid_Optim_with_Cosine_Scheduler(
            self.model,
            total_steps=TrainParams.epochs*len(self.trainer.datamodule.train_dataloader()),
            warmup_steps=TrainParams.epochs*len(self.trainer.datamodule.train_dataloader())//10,
            Optim=HybridOptim,
            OptimHParams=OptimHParams)

        return [optimizer.opt1,optimizer.opt2]
    
    def model_info(self):
        ''' Prints the model summary and the number of parameters in the model.'''
        summary(self.model,input_size=(1,self.config.cwl))

def run_training(model,DataModule):
    ''' Runs the training loop for the model using the Lightning Trainer.
    Args:
        model (Train_Model Class Object): The Train_Model class object containing the GPT model and the training configuration.
        DataModule (DataModule Class Object): The DataModule class object containing the train and validation dataloaders.
    Returns:
        trainer (Trainer Class Object): The trained Lightning Trainer object.
    '''
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        max_epochs=TrainParams.epochs,
        val_check_interval=10,
        log_every_n_steps=50,
        enable_progress_bar=True,
        accumulate_grad_batches=TrainParams.grad_steps,
        gradient_clip_val=1.0

    )
    trainer.fit(model,DataModule)
    return trainer

if __name__=="__main__":
    model=GPT(GPTConfig)
    Datamodule=DataModule(file_path="tokenized_data.npy",train_val_split=0.97)
    run_training(model,Datamodule)

