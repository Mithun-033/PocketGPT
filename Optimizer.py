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



        


    