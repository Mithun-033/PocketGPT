import math
import torch.optim as optim
from HyperParam_Classes import OptimHParams


class HybridOptim():
    ''' 
    Class Module for a hybrid optimizer consisting of AdamW and Muon for faster convergence

    Muon:
        -> Q,K,V Layers
        -> Attention Projection
        -> Up/down MLP projection
        
    AdamW:
        -> RMSNorm
        -> Bias
        -> Embeddings
        -> LM - Head

    Basically any param with dim>=2 --> Muon (Other than embeddings and lm_head),
    While params with dim<2 --> AdamW.
    '''
    def __init__(self,model):
        '''
        Initialising the class.
        
        Args:
            Model (GPT Class Object)
        
        Seperates the parameters into two groups and initialses the optimizers.
        '''
        assert hasattr(optim,"Muon") , "Update torch to 2.10+ to use torch.optim.Muon"

        self.AdamW=[]
        self.Muon=[]

        for name,param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed" in name or "lm_head" in name:
                self.AdamW.append(param)
            elif param.ndim>=2:
                self.Muon.append(param)
            else:
                self.AdamW.append(param)
        
        self.opt1=optim.AdamW(
            self.AdamW,
            lr=OptimHParams.lr,
            weight_decay=OptimHParams.weight_decay,
            fused=True
        )

        self.opt2=optim.Muon(
            self.Muon,
            lr=OptimHParams.lr,
            weight_decay=OptimHParams.weight_decay,
            adjust_lr_function="match_rms_adamw"
        )

    def Count(self) -> None:
        '''Prints the count of parameters in each optimizer'''

        print("-------------------------------------")
        print(f'AdamW Param Count :{len(self.AdamW)}')
        print(f'Muon Param Count :{len(self.Muon)} ')
        print("-------------------------------------")

    def zero_grad(self) -> None:
        ''' Clears the Gradient of previous epoch'''

        self.opt1.zero_grad()
        self.opt2.zero_grad()

    def step(self) -> None:
        ''' Updates the weight & biases'''

        self.opt1.step()
        self.opt2.step()

class Hybrid_Optim_with_Cosine_Scheduler():
    ''' Class module for a hybrid optimizer with a cosine annealing learning rate scheduler with warmup.'''
    def __init__(self,model,Optim,OptimHParams,total_steps,warmup_steps):
        ''' 
        Initialising the class.

        Args:
            model (GPT Class Object)
            Optim (HybridOptim Class)
            OptimHParams (OptimHParams Dataclass Object)
            total_steps (int): Total number of training steps (epochs*steps_per_epoch)
            warmup_steps (int): Number of warmup steps for learning rate scheduler
        '''

        self.optim=Optim(model)
        self.inital_lr=OptimHParams.lr
        self.final_lr=OptimHParams.final_lr
        self.curr_lr=self.inital_lr
        self.total_steps=total_steps
        self.warmup_steps=warmup_steps
        self.current_step=0

    def step(self):
        ''' Updates the learning rate according to the cosine annealing schedule and then calls the step function of the optimizer.'''
        if self.current_step<self.warmup_steps:
            self.curr_lr=self.inital_lr*self.current_step/self.warmup_steps
        else:
            self.curr_lr=self.final_lr+(self.curr_lr-self.final_lr)*((1+math.cos((math.pi*(self.current_step+1))/self.total_steps))/(1+math.cos(math.pi*self.current_step/self.total_steps)))
                        
        for param_group in self.optim.opt1.param_groups:
            param_group["lr"]=self.curr_lr
        
        for param_group in self.optim.opt2.param_groups:
            param_group["lr"]=self.curr_lr

        self.current_step+=1
        self.optim.step()

    def zero_grad(self):
        ''' Clears the Gradient of previous batch'''
        self.optim.zero_grad()






        


    