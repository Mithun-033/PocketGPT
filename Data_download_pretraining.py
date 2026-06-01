import multiprocessing
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

tok=Tokenizer.from_file("tokenizers/tokenizer_32k.json")

def climbmix_1bil():
    target=1_000_000_000
    count=0
    shards=0

    lst=[]

    ds=load_dataset(
        climbmix_path,
        streaming=True,
        split="train")
    with tqdm(total=target, desc="ClimbMix 1bil", unit="Tokens", mininterval=0.1,miniters=1) as pbar:
        for row in ds:
            tokenised=tok.encode(row)
            batch_count=len(tokenised.ids)
            count+=batch_count

            pbar.update(batch_count+2)
            lst.append([tok.bos_token]+tokenised.ids+[tok.eos_token])

            if count>=100_000_000:
                lst=np.array(lst,dtype=np.int16)
                np.save(os.path.join(DATA_DIR,f"shard_{shards+1:02d}.npy"),lst)
                lst=[]
                count=0
                shards+=1

            if shards>=10:
                break



    
    