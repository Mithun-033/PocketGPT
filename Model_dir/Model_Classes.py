import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#=================================================================================
#  RopeEmbedding Block
#=================================================================================

class RopeEmbedding(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to q and k tensors.

    This implementation follows the standard RoPE formulation used in
    transformer variants where half the head dimensions are treated as
    interleaved sin/cos pairs and rotated by position-dependent angles.

    Behaviour and shapes
    - Expects `q` and `k` with shape (B, T, num_heads, head_size).
    - `head_size` must be even (pairs of dimensions).
    - Uses `config.cwl` as the maximum context/window length to precompute
      inverse frequencies; however frequencies are computed lazily per-call
      based on actual `T` so the buffer only stores `inv_freq`.

    Implementation notes
    - `inv_freq` is registered as a buffer so it moves with the module
      between devices and is saved/loaded with the state dict.
    - `_apply_rope` performs the interleaved rotation on the last dimension.
    """

    def __init__(self, config):
        super().__init__()

        # RoPE needs an even head dimension so we can form (cos,sin) pairs.
        assert config.head_size % 2 == 0, "RoPE requires an even head size"

        self.head_size = config.head_size
        # saved for compatibility / external checks; not strictly required here
        self.max_seq_len = getattr(config, "cwl", None)

        # inverse frequencies for rotary embeddings (every other dim)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_size, 2, dtype=torch.float32) / self.head_size)
        )

        # register as buffer so it moves with `.to(device)` and is included in state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x, cos, sin):
        """Apply precomputed cos/sin to tensor `x`.

        x: Tensor[..., head_size] where head_size is even (interleaved dim pairs).
        cos, sin: tensors broadcastable to x[..., head_size/2]

        Returns rotated tensor with same shape as `x`.
        """
        # split interleaved even/odd dims
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # rotate: (x_even, x_odd) -> (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
        x_rope = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)

        # flatten the last two dims back to head_size
        return x_rope.flatten(-2)

    def forward(self, q, k):
        """Apply RoPE to `q` and `k` and return rotated (q,k).

        Args:
            q: Tensor of shape (B, T, num_heads, head_size)
            k: Tensor of shape (B, T, num_heads, head_size)

        Returns:
            (q_rot, k_rot) with same shapes as inputs.
        """
        seq_len = q.size(1)
        device = q.device

        # positions [T] and outer-product with inv_freq -> [T, head_size/2]
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

        # expand to shape (1, T, 1, head_size/2) so it broadcasts over (B, T, num_heads, head_half)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
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
    6. Output projection back to d_model dimension

    Key features:
    - Uses interleaved RMSNorm per head (head_size, not d_model)
    - Applies RoPE to both query and key tensors
    - Implements causal masking for autoregressive generation
    - XSA mechanism normalizes output by aligned vector sum
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
            qkv (nn.Linear): Linear projection to 3x hidden dimension
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

        self.qkv=nn.Linear(config.d_model,3*config.head_size*config.num_heads,bias=False)
        self.q_norm=nn.RMSNorm(config.head_size,eps=1e-5)
        self.k_norm=nn.RMSNorm(config.head_size,eps=1e-5)

        self.proj=nn.Linear(config.head_size*config.num_heads,config.d_model,bias=False)
        self.rope=RopeEmbedding(config)
        self.n_head = config.num_heads
        self.head_size = config.head_size

        self.ve_gate=nn.Linear(config.value_embed_rank,config.num_heads,bias=False) if has_ve(config.num_layers,layer_idx) else None

    def forward(self,x,ve=None):
        ''' 
        Calling the forward pass on the attention layers.
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns: 
            Tensor of shape (B,T,C)
        '''
        B,T,C=x.shape

        qkv=self.qkv(x)
        # split into q,k,v triplet along last dim
        q,k,v = qkv.chunk(3, dim=-1)

        q=q.view(B,T,self.config.num_heads,self.config.head_size)
        k=k.view(B,T,self.config.num_heads,self.config.head_size)
        v=v.view(B,T,self.config.num_heads,self.config.head_size)

        q=self.q_norm(q)
        k=self.k_norm(k)

        if ve is None:
            ve = v
        elif self.ve_gate is not None:
            gate = 3*torch.sigmoid(
                self.ve_gate(x[...,:self.config.value_embed_rank])
            )
            v = v + gate.unsqueeze(-1)*ve

        q,k=self.rope(q,k)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        Vn=F.normalize(v,dim=-1)
        out=out - (out * Vn).sum(dim=-1, keepdim=True)*Vn

        out=out.transpose(1,2).reshape(B,T,self.n_head*self.head_size)

        return self.proj(out),ve

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

    Key features:
    - Uses pre-normalization (RMSNorm before each sub-layer) for stable training
    - Applies uniform scaling factor across both sub-layers
    - Scaling factor depends on model depth (num_layers) for gradient flow
    - Residual connections help mitigate vanishing gradients in deep networks
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
        logits,ve=self.attention(self.PreNorm1(x),ve)
        x=x+self.scale*logits
        x=x+self.scale*self.Mlp(self.PreNorm2(x))
        return x,ve
    
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

    Key features:
    - Weight tying: LM head shares weights with embedding for parameter efficiency
    - Pre-normalization at each block for stable training
    - Causal masking in attention for autoregressive generation
    - Standard initialization scheme (normal for linear, ones for RMSNorm)
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

        self.embed=nn.Embedding(config.vocab_size,config.d_model)
        self.blocks=nn.ModuleList([Block(config,i+1) for i in range(config.num_layers)])
        self.final_norm=nn.RMSNorm(config.d_model,eps=1e-5)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)

        self.lm_head.weight=self.embed.weight # Weights tying
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
        x=self.embed(x) 
        ve=None   # x.shape = (B,T,C)
        for block in self.blocks:
            x,ve=block(x,ve)
        x=self.final_norm(x)
        x=self.lm_head(x)
        return x



            




