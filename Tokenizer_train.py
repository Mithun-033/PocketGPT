from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

DATA_DIR="data"
TOK_DIR="tokenizers"

def tokenizer_16k():
    ''' Trains a BPE tokenizer with a vocab size of 16,000 using the climbmix, mine_q_a and mine_wiki datasets. 
    The trained tokenizer is saved as tokenizer_16k.json in the current directory. '''

    tok=Tokenizer(BPE())
    tok.pre_tokenizer=Whitespace()
    trainer=BpeTrainer(
        vocab_size=16_384,
        special_tokens=["<PAD>","<UNK>","<BOS>","<EOS>"],
        show_progress=True,
    )
    tok.train(
        files=[
            os.path.join(DATA_DIR,"climbmix.txt"),
            os.path.join(DATA_DIR,"mine_q_a.txt"),
            os.path.join(DATA_DIR,"mine_wiki.txt")
        ],
        trainer=trainer
    )
    tok.save(os.path.join(TOK_DIR, "tokenizer_16k.json"))

def tokenizer_32k():
    ''' Trains a BPE tokenizer with a vocab size of 32,000 using the climbmix, mine_q_a and mine_wiki datasets. 
    The trained tokenizer is saved as tokenizer_32k.json in the current directory. '''

    tok=Tokenizer(BPE())
    tok.pre_tokenizer=Whitespace()
    trainer=BpeTrainer(
        vocab_size=32_768,
        special_tokens=["<PAD>","<UNK>","<BOS>","<EOS>"],
        show_progress=True,
    )
    tok.train(
        files=[
            os.path.join(DATA_DIR,"climbmix.txt"),
            os.path.join(DATA_DIR,"mine_q_a.txt"),
            os.path.join(DATA_DIR,"mine_wiki.txt")
        ],
        trainer=trainer
    )
    tok.save(os.path.join(TOK_DIR, "tokenizer_32k.json"))

if __name__=="__main__":
    os.makedirs(TOK_DIR,exist_ok=True)

    print("Training Tokenizer with vocab size 16k...")
    tokenizer_16k()

    print("------------------------------------------------------------------------------")

    print("Training Tokenizer with vocab size 32k...")
    tokenizer_32k()
