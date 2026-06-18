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

def finetune(Model,epochs=2):
    tp=TrainParams()
    gp=Config()

    model=Model.to(device)
    model=torch.compile(model).to(device)

    optimizer=get_optimizer(model,tp,gp)
    loss_fn=nn.CrossEntropyLoss()

    val_dataloader=None

    with tqdm(total=2_600_000_000, desc="Training", unit="Tokens") as pbar:
        opt_steps=torch.load("optimizer_checkpoint.pt",map_location=device)["step"]
        batch_count=0
        for i in range(1):
            file_path=...
            if val_dataloader is None:
                train_dataloader,val_dataloader=get_dataloaders(gp, tp, file_path)
            else:
                train_dataloader,_=get_dataloaders(gp, tp, file_path)

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
                    with open("train_log.json","a") as f:                        
                        json.dump({
                            "step": opt_steps,
                            "train_loss": loss_sum/20,
                            "toks_per_sec": (20*tp.grad_batches*gp.cwl)/time_taken
                        },f)                        
                        f.write("\n")

                    pbar.set_postfix({
                        "train_loss": f"{loss_sum/20:.4f}"
                    })
                    
                    loss_sum=0
                    start=time.time()

                if opt_steps > 0 and opt_steps % 100 == 0 and batch_count == 0:
                    val_loss_sum=0
                    val_batch_count=0
                    
                    if opt_steps % 200 == 0:
                        torch.save(model.state_dict(),"model_checkpoint.pt")
                        torch.save({"optimizer_state_dict": optimizer.state_dict(),
                                "step": opt_steps},"optimizer_checkpoint.pt")

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

                    pbar.set_postfix({
                        "val_loss": f"{val_loss_sum/val_batch_count:.4f}"})
                    
                    with open("val_log.json","a") as f:
                        json.dump({
                            "step": opt_steps,
                            "val_loss": val_loss_sum/val_batch_count
                        },f)
                        f.write("\n")
                    model.train()

        torch.save(model.state_dict(),"final_model.pt")   
            
if __name__ == "__main__":
    state_dict=torch.load("model_checkpoint.pt",map_location=device)

    new_state_dict={}

    for k,v in state_dict.items():
        if k.startswith("_orig_mod."):
            k=k[len("_orig_mod."):]
        new_state_dict[k]=v

    model=GPT(Config()).to(device)
    model.load_state_dict(new_state_dict)

    summary(model,input_size=(1,Config().cwl),dtypes=[torch.long])

    finetune(model, epochs=2)