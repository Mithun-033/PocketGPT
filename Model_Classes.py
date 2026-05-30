import torch.nn as nn
import torch.nn.functional as F
import math

#=================================================================================
#  RopeEmbedding Block
#=================================================================================

class RopeEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        ...

#=================================================================================
#  Multi-Head Attention Block
#=================================================================================

class MultiHeadAttention(nn.Module):
    ''' Class definition of Multi-Head Attention with RMSNorm, RoPE, 
    scaled dot product attention and XSA 
    '''
    def __init__(self,config):
        '''
        Initialising the Multi-Head Attention.
        Args:
            config :-
                -> d_model (int)
                -> head_size (int)
                -> num_heads (int)
        '''
        super().__init__()

        self.config=config
        assert config.head_size*config.num_heads==config.d_model," Dims don't match, check Config params "

        self.qkv=nn.Linear(config.d_model,3*config.head_size*config.num_heads,bias=False)
        self.q_norm=nn.RMSNorm(config.head_size,eps=1e-5)
        self.k_norm=nn.RMSNorm(config.head_size,eps=1e-5)

        self.proj=nn.Linear(config.head_size*config.num_heads,config.d_model,bias=False)
        self.rope=RopeEmbedding()

    def forward(self,x):
        ''' 
        Calling the forward pass on the attention layers.
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns: 
            Tensor of shape (B,T,C)
        '''
        B,T,C=x.shape

        qkv=self.qkv(x)
        q,k,v=qkv.split(self.config.head_size*self.config.num_heads,dim=-1)

        q=q.view(B,T,self.config.num_heads,self.config.head_size)
        k=k.view(B,T,self.config.num_heads,self.config.head_size)
        v=v.view(B,T,self.config.num_heads,self.config.head_size)

        q=self.q_norm(q)
        k=self.k_norm(k)

        q,k=self.rope(q,k)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        out=F.scaled_dot_product_attention(q,k,v,causal=True)
        Vn=F.normalize(v,dim=-1)
        out=out - (out * Vn).sum(dim=-1, keepdim=True)*Vn

        out=out.transpose(1,2).reshape(B,T,self.n_head*self.head_size)

        return self.proj(out)

#=================================================================================
#  MLP Layer
#=================================================================================

class Mlp(nn.Module):
    ''' 
    A Multi-Layer Perceptron class definition with Up and Down projecton linear layers 
    and ReLU square activation.
    '''
    def __init__(self,config):
        ''' 
        Initialising Mlp.

        Args :
            Config DataClass :-
                -> d_model (int)
                -> hidden_size (int)
        '''
        self.up_proj=nn.Linear(config.d_model,config.hidden,bias=False)
        self.down_proj=nn.Linear(config.hidden,config.d_model,bias=False)


    def forward(self,x):
        '''
        Calling the forward on Linear and activation functions
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns:
            Tensor of shape (B,T,C)
        '''

        x=self.up_proj(x)
        x=F.relu(x).square()
        return self.down_proj(x)
    
#=================================================================================
#  Transformer Block 
#=================================================================================
    
class Block(nn.Module):
    ''' Class Definition of a Transformer Block '''
    def __init__(self,config):
        '''
        Initialisng the Block.

        Args:
            config :-
                -> d_model (int)
                -> num_layer (int)
        '''

        self.PreNorm1=nn.RMSNorm(config.d_model,eps=1e-5)
        self.attention=MultiHeadAttention(config)
        self.PreNorm2=nn.RMSNorm(config.d_model,eps=1e-5)
        self.Mlp=Mlp(config)

        self.scale=1/(math.sqrt(2*config.num_layer))

    def forward(self,x):
        ''' Calling the PreNorms, attention and MLP layers on input and scaling the outputs before adding to the residual stream
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns:
            Tensor of shape (B,T,C)
        '''
        x=x+self.scale*self.attention(self.PreNorm1(x))
        x=x+self.scale*self.MLP(self.PreNorm2(x))
        return x
    
#=================================================================================
#  GPT Model
#=================================================================================

class GPT(nn.Module):
    ''' Class definition of the GPT model with token embedding, transformer blocks, final layer norm and language modelling head.'''
    def __init__(self,config):
        '''
        Initialising the GPT model.
        Args:
            config :-
                -> d_model (int)
                -> num_layer (int)
                -> head_size (int)
                -> num_heads (int)
                -> hidden (int)
                -> vocab_size (int)
        '''
        super().__init__()

        self.embed=nn.Embedding(config.vocab_size,config.d_model)
        self.blocks=nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.final_norm=nn.RMSNorm(config.d_model,eps=1e-5)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)

        self.lm_head.weight=self.embed.weight # Weights tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ''' Initialisation of weights for different layers.
        Linear and Embedding layers are initialised with normal distribution with mean 0 
        and std 0.02, while RMSNorm layers are initialised with ones.
        '''
        if isinstance(module,nn.Linear) or isinstance(module,nn.Embedding):
            nn.init.normal(mean=0.0,std=0.02)
        if isinstance(module,nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self,x):
        ''' Calling the forward pass on the GPT model.
        Args:
            x (Tensor): Input Tensor of shape (B,T)
        Returns:
            Tensor of shape (B,T,vocab_size)
        '''
        x=self.embed(x)    # x.shape = (B,T,C)
        for block in self.blocks:
            x=block[x]
        x=self.final_norm(x)
        x=self.lm_head(x)
        return x



            


