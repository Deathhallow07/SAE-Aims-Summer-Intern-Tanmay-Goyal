import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer

from common import MODEL_NAME, HOOK_NAME, MAX_LENGTH, MAX_ACTIVATIONS, DEVICE, save_json


os.makedirs("acts", exist_ok=True)

SAVE_ACTS = "acts/pretrained_code_acts.pt"
SAVE_TOKENS = "acts/pretrained_code_tokens.json"

MAX_CODE_SAMPLES = 3000


def code_text_iterator(dataset, max_items):
    count = 0

    for item in dataset:
        code = item.get("func_code_string", None)

        if code is None:
            code = item.get("whole_func_string", None)

        if code is None:
            continue

        if not isinstance(code, str):
            continue

        if len(code.strip()) < 50:
            continue

        yield code

        count += 1

        if count >= max_items:
            break


print("device:", DEVICE)

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=DEVICE
)

model.eval()

for p in model.parameters():
    p.requires_grad = False

code_dataset = load_dataset(
    "claudios/code_search_net",
    "python",
    split="train",
    streaming=True
)

code_texts = list(code_text_iterator(code_dataset, MAX_CODE_SAMPLES))

all_acts = []
all_token_strs = []

with torch.no_grad():
    for code in tqdm(code_texts):
        tokens = model.to_tokens(
            code,
            truncate=True,
            prepend_bos=True
        ).to(DEVICE)

        tokens = tokens[:, :MAX_LENGTH]

        logits, cache = model.run_with_cache(tokens)

        acts = cache[HOOK_NAME]
        acts = acts.reshape(-1, acts.shape[-1])

        token_strs = model.to_str_tokens(tokens[0])

        acts = acts[1:]
        token_strs = token_strs[1:]

        all_acts.append(acts.detach().cpu().float())
        all_token_strs.extend(token_strs)

        total = sum(x.shape[0] for x in all_acts)

        if total >= MAX_ACTIVATIONS:
            break

activations = torch.cat(all_acts, dim=0)[:MAX_ACTIVATIONS]
all_token_strs = all_token_strs[:MAX_ACTIVATIONS]

torch.save(activations, SAVE_ACTS)
save_json(SAVE_TOKENS, all_token_strs)

print("saved:", SAVE_ACTS)
print("saved:", SAVE_TOKENS)
print("shape:", activations.shape)
