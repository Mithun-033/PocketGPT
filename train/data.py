import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
import random

climbmix_path="karpathy/climbmix-400b-shuffle"
IF_path="thunder-research-group/SNU_Thunder-synthetic-instruction-following"
IF_path_2="iamketan25/alpaca-instructions-dataset"
IF_path_3="m-a-p/CodeFeedback-Filtered-Instruction"

DATA_DIR="Pre_train_data"
DATA_DIR_IF="IF_data"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"

tok=Tokenizer.from_file("tokenizers_dir/tokenizer_49k_whitespace.json")

#-----------------------------------------------------------------------------------------
# ClimbMix 5 Billion Tokens Dataset
#-----------------------------------------------------------------------------------------

def climbmix_5bil():
    '''
    Preprocesses the ClimbMix dataset and saves it as .npy files in the specified directory. 
    Each file contains a shard of the dataset with a maximum of 200 million tokens. 
    The function tokenizes the text data, adds special tokens for beginning and end of sequence, 
    and saves the tokenized data in batches to manage memory usage.
    '''
    target=5_000_000_000
    count=0
    shard=1

    lst=[]

    ds=load_dataset(
        climbmix_path,
        streaming=True,
        split="train")
    
    with tqdm(total=target, desc="ClimbMix 5bil", unit="Tokens", mininterval=0.1,miniters=1) as pbar:
        for row in ds:
            tokenised=tok.encode(row["text"]).ids
            batch_count=len(tokenised)
            
            count+=batch_count+2
            
            pbar.update(batch_count+2)
            lst.extend([2]+tokenised+[3])

            if count>=target//25:
                np.save(os.path.join(DATA_DIR,f"climbmix_{shard}.npy"),np.array(lst,dtype=np.uint16))
                shard+=1
                lst=[]
                print(f"Saved shard {shard-1} with {count} tokens.")
                count=0
            
            if shard>25:
                break


def intruction_finetune():
    '''
    '''

    ds1=load_dataset(
        IF_path,
        "english",
        streaming=True,
        split="train",
        )
    ds2=load_dataset(
        IF_path_2,
        streaming=True,
        split="train",
        )
    ds3=load_dataset(
        IF_path_3,
        streaming=True,
        split="train"
    )
    
    it1=iter(ds1)
    it2=iter(ds2)
    it3=iter(ds3)

    count1=0
    count2=0
    count3=0

    lst=[]

    target=50_000_000

    with tqdm(total=target, desc="Instruction Finetuning", unit="Tokens", mininterval=0.1,miniters=1) as pbar:
        while True:
            rand=random.random()*10

            if rand>2:
                row=next(it1)
                tokenised=tok.encode("Human: "+row["question"]+" Assistant: "+row["response"]).ids

                while len(tokenised)<128:
                    row=next(it1)
                    tokenised=tok.encode("Human: "+row["question"]+" Assistant: "+row["response"]).ids
                
                count1+=len(tokenised)
                lst.extend([2]+tokenised+[3])
                pbar.update(len(tokenised))

            elif rand>1 and rand<2:
                row=next(it3)
                tokenised=tok.encode("Human: "+row["query"]+" Assistant: "+row["answer"]).ids

                while row["lang"]!="python":
                    row=next(it3)
                    tokenised=tok.encode("Human: "+row["query"]+" Assistant: "+row["answer"]).ids

                count3+=len(tokenised)
                lst.extend([2]+tokenised+[3])
                pbar.update(len(tokenised))

            else:
                row=next(it2)
                tokenised=tok.encode(row["prompt"]+row["chosen"]).ids

                while len(tokenised)<128:
                    row=next(it2)
                    tokenised=tok.encode(row["prompt"]+row["chosen"]).ids
                    
                count2+=len(tokenised)

                lst.extend([2]+tokenised+[3])
                pbar.update(len(tokenised))

            
            if count1+count2+count3>=target:

                np.save(os.path.join(DATA_DIR_IF,"IF_dataset_2.npy"),np.array(lst,dtype=np.uint16))
                print(f"Total from IF Dataset 1: {count1} tokens.")
                print(f"Total from IF Dataset 2: {count2} tokens.")
                print(f"Total from IF Dataset 3: {count3} tokens.")
                break

if __name__=="__main__":
    os.makedirs(DATA_DIR,exist_ok=True)
    print("Starting Preprocessing...")

    print("Downloading Climbmix...")
    climbmix_5bil()

    print("Downloading Instruction Finetuning Data...")
    os.makedirs(DATA_DIR_IF,exist_ok=True)
    intruction_finetune()