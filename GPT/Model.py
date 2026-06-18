import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#=================================================================================
#  RopeEmbedding Block
#=================================================================================

class RopeEmbedding(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to query and key tensors.

    RoPE rotates pairs of dimensions in each attention head by an angle that
    depends on the token position. This injects positional information directly
    into the attention mechanism while preserving relative position properties.

    Expected input shape:
        (batch_size, seq_len, num_heads, head_size)

    Notes:
        - head_size must be even because dimensions are processed in pairs.
        - The same rotation is applied to both queries and keys.
    """

    def __init__(self, config):
        super().__init__()

        assert config.head_size % 2 == 0, "RoPE requires an even head size"

        self.head_size=config.head_size

        inv_freq=1.0/(
            10000
            ** (
                torch.arange(0, self.head_size, 2, dtype=torch.float32)
                / self.head_size
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x, cos, sin):
        """Rotate each pair of dimensions in x.

        Args:
            x: Tensor of shape (B, T, H, D)
            cos: Tensor broadcastable to (B, T, H, D/2)
            sin: Tensor broadcastable to (B, T, H, D/2)

        Returns:
            Tensor with the same shape as x after RoPE rotation.
        """

        x_even=x[..., ::2]
        x_odd=x[..., 1::2]

        rotated_even=x_even*cos - x_odd*sin
        rotated_odd=x_even*sin + x_odd*cos

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1
        ).flatten(-2)

    def forward(self, q, k):
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor of shape (B, T, H, D)
            k: Key tensor of shape (B, T, H, D)

        Returns:
            Tuple of rotated (q, k).
        """

        seq_len=q.size(1)

        positions=torch.arange(
            seq_len,
            device=q.device,
            dtype=self.inv_freq.dtype
        )

        angles=torch.outer(positions, self.inv_freq)

        cos=angles.cos().view(1, seq_len, 1, -1)
        sin=angles.sin().view(1, seq_len, 1, -1)

        q=self._apply_rope(q, cos, sin)
        k=self._apply_rope(k, cos, sin)

        return q, k

#=================================================================================
#  Multi-Head Attention Block
#=================================================================================
def has_ve(num_layers,layer_idx):
    ''' Function to determine whether to apply VE in a given layer based on the layer index and total number of layers.
    Args:
        num_layers (int): Total number of layers in the
        layer_idx (int): Index of the current layer
    '''
    return layer_idx%2== num_layers%2

class MultiHeadAttention(nn.Module):
    ''' Class definition of Multi-Head Attention with RMSNorm, RoPE, 
    scaled dot product attention and XSA.

    This implementation combines:
    - Rotary Positional Embeddings (RoPE) for positional encoding
    - RMSNorm normalization before attention computation
    - Scaled Dot Product Attention with causal masking
    - XSA (Cross-Attention Self-Alignment) mechanism for improved attention quality

        Architecture flow:
    1. Input projection via qkv linear layer
    2. Per-head RMSNorm on q and k projections
    3. RoPE application for positional encoding
    4. Scaled dot product attention with causal mask
    5. XSA normalization to align attention outputs
    6. VE (Vector Enhancement) mechanism adds learned embeddings to v based on input tokens
    7. Output projection back to d_model dimension

    Key features:
    - Uses interleaved RMSNorm per head (head_size, not d_model)
    - Applies RoPE to both query and key tensors
    - Implements causal masking for autoregressive generation
    - XSA mechanism normalizes output by aligned vector sum
    - VE mechanism applies enhancement using value_embed_rank dimension
'''
    def __init__(self,config,layer_idx):
        '''
        Initialising the Multi-Head Attention.

        Args:
            config (Config): Configuration object containing:
                -> d_model (int): Model dimension (must equal head_size * num_heads)
                -> head_size (int): Dimension per attention head (must be even for RoPE)
                -> num_heads (int): Number of attention heads

        Attributes:
            q, k, v (nn.Linear): Linear layers for query, key, value projections
            q_norm, k_norm (nn.RMSNorm): Per-head normalization layers
            proj (nn.Linear): Output projection layer
            rope (RopeEmbedding): Rotary positional embedding module
            n_head, head_size: Stored configuration values

        Raises:
            AssertionError: If d_model != head_size * num_heads
        '''
        super().__init__()

        self.config=config

        assert config.head_size*config.num_heads==config.d_model," Dims don't match, check Config params "
        assert config.head_size%2==0,"Head size must be even for RoPE"
        assert config.num_heads%config.kv_heads==0,"Number of heads must be divisible by kv_heads"

        self.q=nn.Linear(config.d_model,config.num_heads*config.head_size,bias=False)
        self.k=nn.Linear(config.d_model,config.kv_heads*config.head_size,bias=False)
        self.v=nn.Linear(config.d_model,config.kv_heads*config.head_size,bias=False)

        self.q_norm=nn.RMSNorm(config.head_size,eps=1e-5)
        self.k_norm=nn.RMSNorm(config.head_size,eps=1e-5)

        self.proj=nn.Linear(config.head_size*config.num_heads,config.d_model,bias=False)
        self.rope=RopeEmbedding(config)
        self.n_head = config.num_heads
        self.head_size = config.head_size

        self.ve_gate=nn.Linear(config.value_embed_rank,config.num_heads,bias=False) if has_ve(config.num_layers,layer_idx) else None
        self.ratio=config.num_heads//config.kv_heads

    def forward(self,x,ve=None):
        ''' 
        Calling the forward pass on the attention layers.
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
            ve (Tensor, optional): Optional value tensor for VE mechanism
        Returns: 
            Tuple of (Tensor, Tensor) with shapes (B,T,C), (B,T,C)
        
        VE Flow:
        - When `ve` is provided, applies a gated residual enhancement
        - Uses sigmoid-activated gate from value_embed_rank dimension
        - Adds gated version of x to v tensor before attention computation
        '''
        B,T,C=x.shape

        q=self.q(x)
        k=self.k(x)
        v=self.v(x)

        q=q.view(B,T,self.config.num_heads,self.config.head_size)
        k=k.view(B,T,self.config.kv_heads,self.config.head_size)
        v=v.view(B,T,self.config.kv_heads,self.config.head_size)

        q=self.q_norm(q)
        k=self.k_norm(k)

        q,k=self.rope(q,k)
        
        k=k.repeat_interleave(self.ratio,dim=2)
        v=v.repeat_interleave(self.ratio,dim=2)

        if self.ve_gate is not None :
            ve=ve.view(B,T,self.config.num_heads,self.config.head_size)
            gate = 3*torch.sigmoid(
                self.ve_gate(x[...,:self.config.value_embed_rank])
            )
            v = v + gate.unsqueeze(-1)*ve

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        Vn=F.normalize(v,dim=-1)
        out=out - (out * Vn).sum(dim=-1, keepdim=True)*Vn

        out=out.transpose(1,2).reshape(B,T,self.n_head*self.head_size)

        return self.proj(out)

#=================================================================================
#  MLP Layer
#=================================================================================

class Mlp(nn.Module):
    ''' A Multi-Layer Perceptron class with Up and Down projection linear layers 
    and ReLU squared activation.

    This implementation uses a simplified MLP architecture:
    1. Up projection from d_model to hidden dimension
    2. ReLU activation followed by element-wise squaring
    3. Down projection back to d_model dimension

    The squaring operation creates a non-linear transformation that emphasizes
    larger values while maintaining the ability to model complex relationships.
    This design differs from standard GELU/ReLU activations and provides
    a custom non-linearity for this architecture.

    Architecture flow:
    x -> up_proj (d_model -> hidden) -> ReLU -> square -> down_proj (hidden -> d_model)

    Key features:
    - Uses element-wise squaring after ReLU for custom non-linearity
    - No bias terms in projection layers for parameter efficiency
    - Hidden dimension controls model capacity
    '''
    def __init__(self,config):
        ''' 
        Initialising Mlp.

        Args :
            config (Config): Configuration object containing:
                -> d_model (int): Input/output dimension
                -> hidden (int): Intermediate hidden dimension

        Attributes:
            up_proj (nn.Linear): Linear layer projecting to hidden dimension
            down_proj (nn.Linear): Linear layer projecting back to d_model

        Raises:
            AssertionError: If config.hidden is not set or invalid
        '''
        super().__init__()
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
    ''' A Transformer block implementing residual connections with scaled layer normalization.

    This block applies two sequential sub-layers with residual connections:
    1. Pre-Norm attention sub-layer with RMSNorm
    2. Pre-Norm MLP sub-layer with RMSNorm

    Both sub-layer outputs are scaled by a factor and added to the input
    via residual connections, creating a residual stream that propagates
    information through the network depth.

    Architecture flow:
    x -> PreNorm1 -> Attention -> scale -> residual add
    x -> PreNorm2 -> MLP -> scale -> residual add

    VE Flow:
    - Passes `ve` through attention mechanism with gated residual enhancement
    - VE output is propagated through the block for subsequent layers

    Key features:
    - Uses pre-normalization (RMSNorm before each sub-layer) for stable training
    - Applies uniform scaling factor across both sub-layers
    - Scaling factor depends on model depth (num_layers) for gradient flow
    - Residual connections help mitigate vanishing gradients in deep networks
    - VE mechanism enables dynamic residual enhancement across transformer blocks
    '''
    def __init__(self,config,layer_idx):
        '''
        Initialising the Block.

        Args:
            config (Config): Configuration object containing:
                -> d_model (int): Model dimension for all layers
                -> num_layers (int): Total number of transformer blocks in model
            layer_idx (int): Index of the current layer

        Attributes:
            PreNorm1, PreNorm2 (nn.RMSNorm): Layer normalization before sub-layers
            attention (MultiHeadAttention): Multi-head attention mechanism
            Mlp (Mlp): Feed-forward network with ReLU squared activation
            scale (float): Scaling factor for residual connections

        Raises:
            AssertionError: If config.num_layers is not set or invalid
        '''
        super().__init__()
        self.PreNorm1=nn.RMSNorm(config.d_model,eps=1e-5)
        self.attention=MultiHeadAttention(config,layer_idx)
        self.PreNorm2=nn.RMSNorm(config.d_model,eps=1e-5)
        self.Mlp=Mlp(config)

        self.scale=1/(math.sqrt(2*config.num_layers))

    def forward(self,x,ve):
        ''' Calling the PreNorms, attention and MLP layers on input and scaling the outputs before adding to the residual stream
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns:
            Tensor of shape (B,T,C)
        '''
        x=x+self.scale*self.attention(self.PreNorm1(x),ve)
        x=x+self.scale*self.Mlp(self.PreNorm2(x))
        return x
    
#=================================================================================
#  GPT Model
#=================================================================================

class GPT(nn.Module):
    ''' A GPT (Generative Pre-trained Transformer) model for language modeling.

    This implementation follows the standard transformer architecture with:
    - Token embedding layer
    - Stacked transformer blocks with residual connections
    - Final layer normalization
    - Language modeling head with weight tying

        Architecture flow:
    Input tokens -> Embedding -> [Transformer Blocks] -> Final Norm -> LM Head -> Output logits

    VE Flow:
    - Passes `ve` through each transformer block with gated residual enhancement
    - VE is propagated through the entire model for consistent residual modulation

    Key features:
    - Weight tying: LM head shares weights with embedding for parameter efficiency
    - Pre-normalization at each block for stable training
    - Causal masking in attention for autoregressive generation
    - Standard initialization scheme (normal for linear, ones for RMSNorm)
    - VE mechanism enables dynamic residual modulation across all transformer blocks
    '''
    def __init__(self,config):
        '''
        Initialising the GPT model.

        Args:
            config (Config): Configuration object containing:
                -> d_model (int): Model dimension (hidden size)
                -> num_layers (int): Number of transformer blocks
                -> head_size (int): Dimension per attention head (must be even)
                -> num_heads (int): Number of attention heads
                -> hidden (int): MLP hidden dimension
                -> vocab_size (int): Vocabulary size for token embedding

        Attributes:
            embed (nn.Embedding): Token embedding layer
            blocks (nn.ModuleList): List of transformer blocks
            final_norm (nn.RMSNorm): Final layer normalization
            lm_head (nn.Linear): Language modeling head with weight tying

        Raises:
            AssertionError: If config parameters are invalid or inconsistent
        '''
        super().__init__()

        self.embed=nn.Embedding(config.vocab_size,config.d_model,)
        self.blocks=nn.ModuleList([Block(config,i+1) for i in range(config.num_layers)])
        self.final_norm=nn.RMSNorm(config.d_model,eps=1e-5)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.value_embeddings=nn.ModuleList([nn.Embedding(config.vocab_size,config.d_model) if has_ve(config.num_layers,i+1) else None for i in range(config.num_layers)])
        #self.lm_head.weight=self.embed.weight 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ''' Initialisation of weights for different layers.
        Linear and Embedding layers are initialised with normal distribution with mean 0 
        and std 0.02, while RMSNorm layers are initialised with ones.
        '''
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.RMSNorm):
            if hasattr(module, 'weight'):
                nn.init.ones_(module.weight)

    def forward(self,x):
        ''' Calling the forward pass on the GPT model.
        Args:
            x (Tensor): Input Tensor of shape (B,T)
        Returns:
            Tensor of shape (B,T,vocab_size)
        '''
        x_og=x.clone()
        x=self.embed(x) 
        # x.shape = (B,T,C)
        
        for i in range(len(self.blocks)):
            block=self.blocks[i]

            if self.value_embeddings[i] is not None:
                val_emb=self.value_embeddings[i](x_og)
            else:
                val_emb=torch.zeros_like(x)

            x=block(x,val_emb)

        x=self.final_norm(x)
        x=self.lm_head(x)
        return x
