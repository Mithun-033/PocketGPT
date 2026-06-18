import torch.optim as optim
import torch

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
    def __init__(self,model,OptimHParams,total_steps):
        '''
        Initialising the class.
        
        Args:
            Model (GPT Class Object)
        
        Seperates the parameters into two groups and initialses the optimizers.
        '''
        assert hasattr(optim,"Muon") , "Update torch to 2.10+ to use torch.optim.Muon"

        self.AdamW_params=[]
        self.Muon_params=[]

        for name,param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed" in name or "lm_head" in name:
                self.AdamW_params.append(param)
            elif param.ndim>=2:
                self.Muon_params.append(param)
            else:
                self.AdamW_params.append(param)

        self.config=OptimHParams()
        self.opt1=optim.AdamW(
            self.AdamW_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            fused=True if torch.cuda.is_available() else False,
            betas=(self.config.betas[0],self.config.betas[1]),
        )
        self.opt2=optim.Muon(
            self.Muon_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            adjust_lr_fn="match_rms_adamw",

        )

        self.scheduler_adam1=optim.lr_scheduler.LinearLR(
            self.opt1,
            start_factor=0.2,
            total_iters=int(0.05 * total_steps),
        )
        self.scheduler_adam2=optim.lr_scheduler.CosineAnnealingLR(
            self.opt1,
            T_max=int(total_steps*0.95),
            eta_min=self.config.final_lr
        )
        self.scheduler_muon1=optim.lr_scheduler.LinearLR(
            self.opt2,
            start_factor=0.2,
            total_iters=int(0.05 * total_steps),
        )
        self.scheduler_muon2=optim.lr_scheduler.CosineAnnealingLR(
            self.opt2,
            T_max=int(total_steps*0.95),
            eta_min=self.config.final_lr
        )
        self.Adamw=optim.lr_scheduler.SequentialLR(
            self.opt1,
            schedulers=[self.scheduler_adam1,self.scheduler_adam2],
            milestones=[int(0.05 * total_steps)]
        )
        self.Muon=optim.lr_scheduler.SequentialLR(
            self.opt2,
            schedulers=[self.scheduler_muon1,self.scheduler_muon2],
            milestones=[int(0.05 * total_steps)]
        )


    def Count(self) -> None:
        '''Prints the count of parameters in each optimizer'''

        print("-------------------------------------")
        print(f'AdamW Param Count :{len(self.opt1.param_groups[0]["params"])}')
        print(f'Muon Param Count :{len(self.opt2.param_groups[0]["params"])} ')
        print("-------------------------------------")

    def zero_grad(self) -> None:
        ''' Clears the Gradient of previous epoch'''

        self.opt1.zero_grad()
        self.opt2.zero_grad()

    def step(self) -> None:
        ''' Updates the weight & biases'''

        self.opt1.step()
        self.opt2.step()
        self.Adamw.step()
        self.Muon.step()

    def state_dict(self) -> dict:
        ''' Returns the state dict of both optimizers and schedulers'''

        return {
            "opt1":self.opt1.state_dict(),
            "opt2":self.opt2.state_dict(),
            "Adamw":self.Adamw.state_dict(),
            "Muon":self.Muon.state_dict()
        }
    
    def load_state_dict(self,state_dict:dict) -> None:
        ''' Loads the state dict of both optimizers and schedulers'''

        self.opt1.load_state_dict(state_dict["opt1"])
        self.opt2.load_state_dict(state_dict["opt2"])
        self.Adamw.load_state_dict(state_dict["Adamw"])
        self.Muon.load_state_dict(state_dict["Muon"])






        


    