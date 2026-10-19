from dataclasses import dataclass

@dataclass 
class Config:
    """Configuration class for model hyperparameters.
    This class stores all the architectural and structural parameters of the GPT model including:
    - cwl (context window length): Maximum sequence length for attention mechanism
    - d_model: Dimension size of transformer embeddings
    - num_layers: Number of transformer blocks/layers in the model
    - head_size: Size of each attention head dimension
    - num_heads: Number of attention heads
    - hidden: Hidden layer size for MLP layers (typically 4x d_model)
    - vocab_size: Vocabulary size for token embeddings and output projection
    """
    cwl : int = 1024
    d_model : int = 1024
    num_layers : int = 20
    head_size : int = 64
    num_heads : int = 16
    kv_heads: int = 4
    hidden : int = d_model*4
    vocab_size : int = 49_152
    value_embed_rank : int = 16

@dataclass 
class OptimHParams:
    """Optimization hyperparameters for training the GPT model.
    
    This class stores learning rate and optimization settings including:
    - lr (learning_rate): Initial base learning rate for optimizer
    - weight_decay: L2 regularization strength for weight decay
    - lr_decay: Learning rate decay factor for final learning rate calculation
    """
    lr : float = 5e-4
    weight_decay : int = 0.1
    lr_decay : float = 0.1
    final_lr : float = lr * lr_decay
    betas : tuple = (0.9, 0.95)

@dataclass 
class OptimHParams_FT:
    """Optimization hyperparameters for fine-tuning the GPT model.
    This class is similar to OptimHParams but with a lower learning rate and weight decay suitable for fine-tuning.

    """
    lr : float = 3e-5
    weight_decay : int = 0.01
    lr_decay : float = 0.1
    final_lr : float = lr * lr_decay

@dataclass
class TrainParams:
    """
    epochs (int): Number ofi traning epochs to run
    batch_size (int): Number of samples per training batch
    grad_batches (int): Gradient accumulation batches for effective batch size
    num_workers (int): Number of worker processes for data loading
    pre_fetch_factor (int): Factor for prefetching data during training
    
    Note: The effective batch size is calculated as batch_size * grad_batches.
    """
    epochs : int = 1
    batch_size : int = 32
    grad_batches : int = 512
    num_workers : int = 6
    pre_fetch_factor : int = 3
