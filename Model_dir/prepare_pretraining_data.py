import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer

climbmix_path="karpathy/climbmix-400b-shuffle"
minecraft_path="lparkourer10/minecraft-wiki"
minecraft_path2="minhaozhang/minecraft-question-answer-630k"

DATA_DIR="Pre_train_data"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"

tok=Tokenizer.from_file("tokenizers_dir/tokenizer_32k.json")

#-----------------------------------------------------------------------------------------
# ClimbMix 2 Billion Tokens Dataset
#-----------------------------------------------------------------------------------------

def climbmix_2bil():
    '''
    Preprocesses the ClimbMix dataset and saves it as .npy files in the specified directory. 
    Each file contains a shard of the dataset with a maximum of 100 million tokens. 
    The function tokenizes the text data, adds special tokens for beginning and end of sequence, 
    and saves the tokenized data in batches to manage memory usage.
    '''
    target=2_000_000_000
    count=0
    shard=1

    lst=[]

    ds=load_dataset(
        climbmix_path,
        streaming=True,
        split="train")
    
    with tqdm(total=target, desc="ClimbMix 2bil", unit="Tokens", mininterval=0.1,miniters=1) as pbar:
        for row in ds:
            tokenised=tok.encode(row["text"]).ids
            batch_count=len(tokenised)
            
            count+=batch_count+2
            
            pbar.update(batch_count+2)
            lst.extend([2]+tokenised+[3])

            if count>=target//10:
                np.save(os.path.join(DATA_DIR,f"climbmix_{shard}.npy"),np.array(lst,dtype=np.uint16))
                shard+=1
                lst=[]
                print(f"Saved shard {shard-1} with {count} tokens.")
                count=0
            
            if shard>10:
                break

#-----------------------------------------------------------------------------------------
# Minecraft Wiki Dataset
#-----------------------------------------------------------------------------------------

def mine_wiki():
    '''
    Preprocesses the Minecraft Wiki dataset and saves it as a .npy file in the specified directory.
    The function tokenizes the text data, adds special tokens for beginning and end of sequence,
    and saves the tokenized data in a single .npy file.
    '''

    lst=[]
    count=0

    ds=load_dataset(
        minecraft_path,
        streaming=True,
        split="train")

    with tqdm(desc="Mine_wiki", unit="Tokens", mininterval=0.1,miniters=1) as pbar:    
        for row in ds:
            tokenized=tok.encode("Question :"+row["question"]+"\n"+"Answer :"+row["answer"]).ids
            batch_count=len(tokenized)
            count+=batch_count+2
            lst.extend([2]+tokenized+[3])

            pbar.update(batch_count+2)
        
    np.save(os.path.join(DATA_DIR,"mine_wiki.npy"),np.array(lst,dtype=np.uint16))
    print(f"Total Tokens (Mine-wiki): {count}")

#-----------------------------------------------------------------------------------------
# Minecraft - Q&A Dataset
#-----------------------------------------------------------------------------------------

def mine_qa():
    '''
    Preprocesses the Minecraft Q&A dataset and saves it as a .npy file in the specified directory.
    The function tokenizes the text data, adds special tokens for beginning and end of sequence,
    and saves the tokenized data in a single .npy file.
    '''

    lst=[]
    count=0

    ds=load_dataset(
        minecraft_path2,
        streaming=True,
        split="train")
    
    with tqdm(desc="Mine_q_a", unit="Tokens", mininterval=0.1,miniters=1) as pbar:    
        for row in ds:
            tokenized=tok.encode("Question :"+row["question"]+"\n"+"Answer :"+row["answer"]).ids
            batch_count=len(tokenized)
            count+=batch_count+2
            lst.extend([2]+tokenized+[3])

            pbar.update(batch_count+2)


    np.save(os.path.join(DATA_DIR,"mine_qa.npy"),np.array(lst,dtype=np.uint16))
    print(f"Total Tokens (Mine-QA): {count}")

if __name__=="__main__":
    os.makedirs(DATA_DIR,exist_ok=True)
    print("Starting Preprocessing...")

    # print("Downloading Climbmix...")
    # climbmix_2bil()

    print("Downloading Minecraft Wiki...")
    mine_wiki()

    print("Downloading Minecraft QA...")
    mine_qa()

