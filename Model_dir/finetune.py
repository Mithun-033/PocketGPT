import torch
import torch.nn as nn
from torchinfo import summary
from Optimizer import HybridOptim
from Model_Classes import GPT

from train import get_dataloaders
from HyperParam_Classes import Config, OptimHParams_FT, TrainParams
import time
import json 
from tqdm import tqdm
import warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)

def get_optimizer(model, tp, gp, epochs):
    '''Initialises the Hybrid Optimizer with Cosine Scheduler.

    Args:
        op (OptimHParams): Optimization hyperparameters including learning rate and weight decay.
        model (GPT): The GPT model instance for which the optimizer is being created.
        tp (TrainParams): Training parameters including batch size and number of workers.
        gp (Config): Model configuration parameters including context window length and embedding dimensions.
    '''
    

    optimizer = HybridOptim(
        model=model,
        OptimHParams=OptimHParams_FT, 
        total_steps= (47509085 + 8026886)*epochs // (tp.grad_batches * gp.cwl)
    )
    return optimizer

def finetune(model,epochs=2):
    model.to(device)
    model=torch.compile(model).to(device)

    gp=Config()
    tp=TrainParams()
    optimizer=get_optimizer(model, tp, gp, epochs)
    loss_fn=nn.CrossEntropyLoss()

    opt_steps=0
    with tqdm(total=(47509085 + 8026886)*epochs, desc="Fine-tuning", unit="Tokens") as pbar:
        for epoch in range(epochs):
            file_paths=["../Pre_train_data/mine_qa.npy","../Pre_train_data/mine_wiki.npy"]

            for file in file_paths:
                train_dataloader,val_dataloader=get_dataloaders(gp, tp, file)

                batch_count=0
                loss_sum=0

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
                            "step": opt_steps
                        },"checkpoint_finetuned.pt")

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
                        with open("training_log_finetuned.json","a") as f:
                            json.dump({
                                "step": opt_steps,
                                "val_loss": val_loss_sum/val_batch_count
                            },f)
                            f.write("\n")
            torch.save(model.state_dict(),f"finetuned_model_{epoch}.pt")
            
if __name__ == "__main__":
    gp=Config()
    tp=TrainParams()

    print("Loading model...")
    model=GPT(gp)
    model.load_state_dict(torch.load("checkpoint.pt")["model"])
    print("Model loaded successfully.")

    summary(model, input_size=(tp.batch_size, gp.cwl), dtypes=[torch.long], device=device)

    finetune(model, epochs=2)