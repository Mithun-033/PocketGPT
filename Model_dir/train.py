import time
from tqdm import tqdm
import torch
import torch.nn as nn

from Model_Classes import GPT
from HyperParam_Classes import TrainParams, Config, OptimHParams
from Optimizer import HybridOptim

from DataLoaders import DataModule
from torchinfo import summary
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(config,tp,file_path):
    '''Initialises the DataModule and returns the Train and Validation DataLoaders.

    Args:
        config (Config): Configuration object containing hyperparameters including context window length (cwl).
        file_path (str): Path to the data file.

    Returns:
        tuple: A tuple containing (train_dataloader, val_dataloader) where:
            - train_dataloader (DataLoader): The training data loader.
            - val_dataloader (DataLoader): The validation data loader.
    '''
    data_module = DataModule(
        file_path=file_path,
        train_val_split=0.9998,
        num_workers=tp.num_workers,
        pin_memory=True,
        persistent_workers=True,
        batch_size=tp.batch_size,
        pre_fetch_factor=tp.pre_fetch_factor,
        config=config
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module.train_dataloader(), data_module.val_dataloader()

def get_optimizer(model, tp, op, gp):
    '''Initialises the Hybrid Optimizer with Cosine Scheduler.

    Args:
        op (OptimHParams): Optimization hyperparameters including learning rate and weight decay.
        model (GPT): The GPT model instance for which the optimizer is being created.
        tp (TrainParams): Training parameters including batch size and number of workers.
        gp (Config): Model configuration parameters including context window length and embedding dimensions.
    '''
    

    optimizer = HybridOptim(
        model=model,
        OptimHParams=op,
        total_steps=tp.total_steps
    )
    return optimizer

def train():
    '''Main training loop for the GPT model.
    This function initializes the model, optimizer, and data loaders, 
    and then iterates through the training data to perform optimization steps. 
    It also periodically evaluates the model on the validation set and prints training and validation loss.
    '''
    tp=TrainParams()
    gp=Config()
    op=OptimHParams()

    model=GPT(gp).to(device)
    model=torch.compile(model).to(device)
    print(summary(model,input_size=(tp.batch_size,gp.cwl),dtypes=[torch.long]))

    optimizer=get_optimizer(model,tp,op,gp)
    loss_fn=nn.CrossEntropyLoss()

    val_dataloader=None

    with tqdm(total=2_000_000_000, desc="Training", unit="Tokens") as pbar:
        for i in range(10):
            file_path=f"Pre_train_data/climbmix_{i+1}.npy"
            if val_dataloader is None:
                train_dataloader,val_dataloader=get_dataloaders(gp, tp, file_path)
            else:
                train_dataloader,_=get_dataloaders(gp, tp, file_path)

            batch_count=0
            loss_sum=0
            opt_steps=0
            start=time.time()

            model.train()

            for x,y in train_dataloader:
                x=x.to(device,non_blocking=True)
                y=y.to(device,non_blocking=True)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    out=model(x)
                    loss=loss_fn(out.view(-1,gp.vocab_size),y.view(-1))
                    loss=loss/(tp.grad_batches/tp.batch_size)
                    loss.backward()
                    
                loss_sum+=loss.item()
                batch_count+=tp.batch_size
                pbar.update(tp.batch_size*gp.cwl)

                if batch_count>=tp.grad_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_count=0
                    opt_steps+=1

                if opt_steps > 0 and opt_steps % 20 == 0 and batch_count == 0:
                    time_taken=time.time()-start
                    print(f"Loss: {loss_sum/20:.4f}, Time : {time_taken:.2f} seconds, Toks/sec : {(20*tp.grad_batches*gp.cwl)/time_taken:.2f}")
                    loss_sum=0
                    start=time.time()

                if opt_steps > 0 and opt_steps % 100 == 0 and batch_count == 0:
                    val_loss_sum=0
                    val_batch_count=0
                    
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": opt_steps
                    },"checkpoint.pt")

                    model.eval()
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                            for x,y in val_dataloader:
                                x=x.to(device,non_blocking=True)
                                y=y.to(device,non_blocking=True)
                                out=model(x)
                                loss=loss_fn(out.view(-1,gp.vocab_size),y.view(-1))
                                val_loss_sum+=loss.item()
                                val_batch_count+=1

                    print(f"Validation Loss: {val_loss_sum/val_batch_count:.4f}")
                    model.train()
                    with open("training_log.json","a") as f:
                        json.dump({
                            "step": opt_steps,
                            "val_loss": val_loss_sum/val_batch_count
                        },f)
                        f.write("\n")

if __name__=="__main__":
    train()









    
