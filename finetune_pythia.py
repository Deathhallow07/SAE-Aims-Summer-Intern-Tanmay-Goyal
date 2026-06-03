import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from common import MODEL_NAME, MAX_LENGTH, DEVICE


os.makedirs("models", exist_ok=True)

FT_BATCH_SIZE = 2
FT_LR = 1e-5
FT_STEPS = 500
GRAD_ACCUM_STEPS = 4
MAX_CODE_SAMPLES = 3000

SAVE_PATH = "models/pythia160m_code_finetuned.pt"


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

model.train()

code_dataset = load_dataset(
    "claudios/code_search_net",
    "python",
    split="train",
    streaming=True
)

code_texts = list(code_text_iterator(code_dataset, MAX_CODE_SAMPLES))

all_token_batches = []

for code in tqdm(code_texts):
    tokens = model.to_tokens(
        code,
        truncate=True,
        prepend_bos=True
    )

    tokens = tokens[:, :MAX_LENGTH]

    if tokens.shape[1] >= 16:
        all_token_batches.append(tokens.squeeze(0).detach().cpu())

pad_token = model.tokenizer.eos_token_id

padded_batches = []

for tokens in all_token_batches:
    seq_len = tokens.shape[0]

    if seq_len < MAX_LENGTH:
        pad_len = MAX_LENGTH - seq_len
        padding = torch.full((pad_len,), pad_token, dtype=tokens.dtype)
        tokens = torch.cat([tokens, padding], dim=0)
    else:
        tokens = tokens[:MAX_LENGTH]

    padded_batches.append(tokens)

token_tensor = torch.stack(padded_batches)

ft_loader = DataLoader(
    token_tensor,
    batch_size=FT_BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR)

step = 0
optimizer.zero_grad()

progress = tqdm(total=FT_STEPS)

while step < FT_STEPS:
    for batch_tokens in ft_loader:
        batch_tokens = batch_tokens.to(DEVICE)

        logits = model(batch_tokens)

        logits_for_loss = logits[:, :-1, :]
        targets = batch_tokens[:, 1:]

        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_token
        )

        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 20 == 0:
            progress.set_description(f"loss={loss.item() * GRAD_ACCUM_STEPS:.4f}")

        step += 1
        progress.update(1)

        if step >= FT_STEPS:
            break

progress.close()

prompt = "def add_numbers(a, b):\n"
tokens = model.to_tokens(prompt).to(DEVICE)
model.eval()
with torch.no_grad():
    generated = model.generate(
        tokens,
        max_new_tokens=80,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )
print(model.to_string(generated[0]))

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "fine_tune_domain": "python_code",
        "ft_steps": FT_STEPS,
        "ft_lr": FT_LR,
        "max_length": MAX_LENGTH
    },
    SAVE_PATH
)

print("saved:", SAVE_PATH)
