from dataclasses import dataclass

@dataclass 
class Config:
    d_model : int = 768
    num_layers : int = 16
    head_size : int = 64
    num_heads : int = 12
    hidden : int = d_model*4
    vocab_size : int = ...

@dataclass 
class OptimHParams:
    lr : float = 5e-4
    weight_decay : int = 0.1
    lr_decay : float = 0.1
    final_lr : float = lr * lr_decay

@dataclass
class TrainParams:
    epochs : int = 1
    batch_size : int = 16
    grad_steps : int = int(512/batch_size)


    