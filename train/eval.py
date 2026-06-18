import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import argparse
from GPT.Model import GPT
from GPT.Hyperparams import Config
from tokenizers import Tokenizer
import os

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
state_dict=torch.load("/teamspace/studios/this_studio/fine_tuned_model_0.pt",map_location=DEVICE)

new_state_dict={}

for k,v in state_dict.items():
    if k.startswith("_orig_mod."):
        k=k[len("_orig_mod."):]
    new_state_dict[k]=v

model=GPT(Config()).to(DEVICE)
model.load_state_dict(new_state_dict)

model.eval()
tokenizer=Tokenizer.from_file("/teamspace/studios/this_studio/PocketGPT/tokenizers_dir/tokenizer_49k_whitespace.json")

torch.set_float32_matmul_precision("high")

model=torch.compile(
    model,
    mode="reduce-overhead"
)
#####################################################################
# MODEL INTERFACE
#####################################################################


def encode(text):
    return tokenizer.encode(text).ids


@torch.no_grad()
def get_logits(ids):
    x=torch.tensor(ids,dtype=torch.long,device=DEVICE).unsqueeze(0)
    logits=model(x)
    return logits[0]


@torch.no_grad()
def logprob(context,continuation):
    full=context+continuation

    full_ids=encode(full)
    ctx_ids=encode(context)

    max_len=Config().cwl

    if len(full_ids)>max_len:
        overflow=len(full_ids)-max_len

        full_ids=full_ids[overflow:]

        if overflow<len(ctx_ids):
            ctx_ids=ctx_ids[overflow:]
        else:
            ctx_ids=[]

    logits=get_logits(full_ids)

    log_probs=F.log_softmax(logits[:-1],dim=-1)

    score=0.0
    count=0

    for i in range(len(ctx_ids),len(full_ids)):
        token=full_ids[i]
        score+=log_probs[i-1,token].item()
        count+=1

    if count==0:
        return -1e9

    return score/count


#####################################################################
# GENERIC MULTIPLE CHOICE
#####################################################################


@torch.no_grad()
def mcq_score(prompt,choices):
    ctx_ids=encode(prompt)

    seqs=[]
    choice_ids_list=[]

    for choice in choices:
        choice_ids=encode(" "+choice)

        full_ids=ctx_ids+choice_ids

        if len(full_ids)>Config().cwl:
            full_ids=full_ids[-Config().cwl:]

        seqs.append(full_ids)
        choice_ids_list.append(choice_ids)

    max_len=max(len(seq) for seq in seqs)

    batch=torch.zeros(
        (len(seqs),max_len),
        dtype=torch.long,
        device=DEVICE
    )

    for i,seq in enumerate(seqs):
        batch[i,:len(seq)]=torch.tensor(
            seq,
            dtype=torch.long,
            device=DEVICE
        )

    logits=model(batch)
    log_probs=F.log_softmax(logits[:,:-1],dim=-1)

    scores=[]

    for b,choice_ids in enumerate(choice_ids_list):
        start=len(seqs[b])-len(choice_ids)

        score=0.0
        count=0

        for j,token in enumerate(choice_ids):
            pos=start+j

            if pos==0:
                continue

            score+=log_probs[b,pos-1,token].item()
            count+=1

        scores.append(score/max(count,1))

    return max(
        range(len(scores)),
        key=lambda i:scores[i]
    )


#####################################################################
# HELLASWAG
#####################################################################


def eval_hellaswag(limit=None):
    # HellaSwag
    ds=load_dataset("Rowan/hellaswag",split="validation")

    correct=0
    total=0

    for row in tqdm(ds):
        pred=mcq_score(row["ctx"],row["endings"])

        if pred==int(row["label"]):
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total



#####################################################################
# WINOGRANDE
#####################################################################


def eval_winogrande(limit=None):
    ds=load_dataset(
        "allenai/winogrande",
        "winogrande_xl",
        split="validation"
    )

    correct=0
    total=0

    for row in tqdm(ds):
        scores=[]

        for option in [row["option1"],row["option2"]]:
            completed=row["sentence"].replace("_",option)
            scores.append(logprob("",completed))

        pred=max(range(len(scores)),key=lambda i:scores[i])

        if pred+1==int(row["answer"]):
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# ARC
#####################################################################


def eval_arc(config,limit=None):
    # ARC
    ds=load_dataset("allenai/ai2_arc",config,split="test")

    correct=0
    total=0

    for row in tqdm(ds):
        choices=row["choices"]["text"]
        labels=row["choices"]["label"]

        pred=mcq_score(row["question"],choices)

        pred_label=labels[pred]

        if pred_label==row["answerKey"]:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# OPENBOOKQA
#####################################################################


def eval_openbookqa(limit=None):
    # OpenBookQA
    ds=load_dataset("allenai/openbookqa","main",split="test")

    correct=0
    total=0

    for row in tqdm(ds):
        pred=mcq_score(
            row["question_stem"],
            row["choices"]["text"]
        )

        label=row["choices"]["label"][pred]

        if label==row["answerKey"]:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# MMLU
#####################################################################


def eval_mmlu(limit=None):
    ds=load_dataset("cais/mmlu","all",split="test")

    correct=0
    total=0

    for row in tqdm(ds):
        pred=mcq_score(
            row["question"],
            row["choices"]
        )

        if pred==row["answer"]:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# RUNNER
#####################################################################


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args=parser.parse_args()

    benchmarks=[
    ("hellaswag",lambda:eval_hellaswag(args.limit)),
    ("winogrande",lambda:eval_winogrande(args.limit)),
    ("arc_easy",lambda:eval_arc("ARC-Easy",args.limit)),
    ("arc_challenge",lambda:eval_arc("ARC-Challenge",args.limit)),
    ("openbookqa",lambda:eval_openbookqa(args.limit)),
    ("mmlu",lambda:eval_mmlu(args.limit)),
]

    results={}

    for name,fn in benchmarks:
        results[name]=fn()
        print(f"{name:20s} {results[name]:.2f}")


if __name__=="__main__":
    main()