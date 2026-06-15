import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer

climbmix_path="karpathy/climbmix-400b-shuffle"

DATA_DIR="Pre_train_data"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"

tok=Tokenizer.from_file("tokenizers_dir/tokenizer_49k_whitespace.json")

#-----------------------------------------------------------------------------------------
# ClimbMix 6 Billion Tokens Dataset
#-----------------------------------------------------------------------------------------

def climbmix_6bil():
    '''
    Preprocesses the ClimbMix dataset and saves it as .npy files in the specified directory. 
    Each file contains a shard of the dataset with a maximum of 200 million tokens. 
    The function tokenizes the text data, adds special tokens for beginning and end of sequence, 
    and saves the tokenized data in batches to manage memory usage.
    '''
    target=6_000_000_000
    count=0
    shard=1

    lst=[]

    ds=load_dataset(
        climbmix_path,
        streaming=True,
        split="train")
    
    with tqdm(total=target, desc="ClimbMix 6bil", unit="Tokens", mininterval=0.1,miniters=1) as pbar:
        for row in ds:
            tokenised=tok.encode(row["text"]).ids
            batch_count=len(tokenised)
            
            count+=batch_count+2
            
            pbar.update(batch_count+2)
            lst.extend([2]+tokenised+[3])

            if count>=target//30:
                np.save(os.path.join(DATA_DIR,f"climbmix_{shard}.npy"),np.array(lst,dtype=np.uint16))
                shard+=1
                lst=[]
                print(f"Saved shard {shard-1} with {count} tokens.")
                count=0
            
            if shard>30:
                break

if __name__=="__main__":
    os.makedirs(DATA_DIR,exist_ok=True)
    print("Starting Preprocessing...")

    print("Downloading Climbmix...")
    climbmix_6bil()

