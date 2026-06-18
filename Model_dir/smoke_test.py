from Model_Classes import GPT
from HyperParam_Classes import Config,OptimHParams
from torchinfo import summary
import torch
from Optimizer import HybridOptim

print("cu"+str(int(float(torch.version.cuda)*10)))

# from tokeknizers import Tokenizer
# tok=Tokenizer.from_file("tokenizers_dir/tokenizer_49k_whitespace.json")

# print(tok.encode("<BOS>").ids)

device="cuda" if torch.cuda.is_available() else "cpu"

model=GPT(Config()).to(device)
#model=torch.compile(model).to(device)

model(torch.randint(0,32786,(1,1024)).to(device))
print("Input passed through the model successfully!")
summary(model,input_size=(1,1024),dtypes=[torch.long],device=device)

opt=HybridOptim(model,OptimHParams,total_steps=6_000_000_000//(1*1024))
opt.Count()
