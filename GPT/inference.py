import torch
import torch.nn.functional as F
from Model_dir.Model import GPT
from Model_dir.Hyperparams import Config
from tokenizers import Tokenizer

device="cuda" if torch.cuda.is_available() else "cpu"

state_dict=torch.load("C:\\Users\\mithu\\Desktop\\Mithun\\MineGPT\\fine_tuned_model_0.pt",map_location=device)

new_state_dict={}

for k,v in state_dict.items():
    if k.startswith("_orig_mod."):
        k=k[len("_orig_mod."):]
    new_state_dict[k]=v

model=GPT(Config()).to(device)
model.load_state_dict(new_state_dict)

model.eval()
print("Model loaded successfully!")

tokenizer=Tokenizer.from_file("C:\\Users\\mithu\\Desktop\\Mithun\\MineGPT\\tokenizers_dir\\tokenizer_49k_whitespace.json")
print("Tokenizer loaded successfully!")

@torch.no_grad()
def generate(
    prompt_ids,
    max_new_tokens=200,
    temperature=0.85,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1,
    repetition_window=128,
    eos_token_id=3
):
    x=torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            logits=model(x)

            if isinstance(logits,tuple):
                logits=logits[0]

            logits=logits[:,-1,:]

            if repetition_penalty!=1.0:
                recent_tokens=x[0,-repetition_window:]
                unique_tokens=torch.unique(recent_tokens)

                token_logits=logits[0,unique_tokens]

                token_logits=torch.where(
                    token_logits>0,
                    token_logits/repetition_penalty,
                    token_logits*repetition_penalty
                )

                logits[0,unique_tokens]=token_logits

            if temperature==0:
                next_token=torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )
            else:
                logits=logits/temperature

                if top_k is not None:
                    v,_=torch.topk(
                        logits,
                        min(top_k,logits.size(-1))
                    )
                    logits[logits<v[:,-1,None]]=-float("inf")

                probs=F.softmax(
                    logits,
                    dim=-1
                )

                if top_p is not None:
                    sorted_probs,sorted_idx=torch.sort(
                        probs,
                        descending=True
                    )

                    cumulative=torch.cumsum(
                        sorted_probs,
                        dim=-1
                    )

                    mask=cumulative>top_p
                    mask[...,1:]=mask[...,:-1].clone()
                    mask[...,0]=False

                    sorted_probs[mask]=0

                    sorted_probs=sorted_probs/(
                        sorted_probs.sum(
                            dim=-1,
                            keepdim=True
                        )
                    )

                    next_token=sorted_idx.gather(
                        -1,
                        torch.multinomial(
                            sorted_probs,
                            1
                        )
                    )
                else:
                    next_token=torch.multinomial(
                        probs,
                        1
                    )

            x=torch.cat(
                [x,next_token],
                dim=1
            )

            if (
                eos_token_id is not None and
                next_token.item()==eos_token_id
            ):
                break

    return x[0].tolist()

while True:
    prompt=input("Enter a prompt: ")
    prompt="Human: "+prompt+" Assistant: "
    prompt_ids=tokenizer.encode(prompt).ids
    length=len(prompt_ids)
    generated_ids=generate(prompt_ids)[length:]
    generated_text=tokenizer.decode(generated_ids)
    print(generated_text)