import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from common import SparseAutoencoder, MODEL_NAME, HOOK_NAME, D_IN, D_SAE, SAE_BATCH_SIZE, SAE_LR, L1_COEFF, SAE_EPOCHS, DEVICE


os.makedirs("models", exist_ok=True)

ACTS_PATH = "acts/finetuned_code_acts.pt"
SAVE_PATH = "models/code_finetuned_pythia160m_layer6_sae.pt"

activations = torch.load(ACTS_PATH)

sae = SparseAutoencoder(d_in=D_IN, d_sae=D_SAE).to(DEVICE)

train_loader = DataLoader(
    TensorDataset(activations),
    batch_size=SAE_BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

for epoch in range(SAE_EPOCHS):
    sae.train()

    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_l0 = 0.0

    for batch in tqdm(train_loader, desc=f"ft sae epoch {epoch + 1}/{SAE_EPOCHS}"):
        x = batch[0].to(DEVICE)

        reconstruction, features = sae(x)

        mse_loss = F.mse_loss(reconstruction, x)
        l1_loss = features.abs().sum(dim=-1).mean()

        loss = mse_loss + L1_COEFF * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sae.normalize_decoder()

        with torch.no_grad():
            l0 = (features > 0).float().sum(dim=-1).mean()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_l1 += l1_loss.item()
        total_l0 += l0.item()

    n = len(train_loader)

    print(
        f"epoch {epoch + 1} | "
        f"loss {total_loss / n:.6f} | "
        f"mse {total_mse / n:.6f} | "
        f"l1 {total_l1 / n:.6f} | "
        f"avg_l0 {total_l0 / n:.2f}"
    )

sae.eval()

with torch.no_grad():
    x = activations[:2048].to(DEVICE)
    reconstruction, features = sae(x)

    final_mse = F.mse_loss(reconstruction, x).item()
    final_l0 = (features > 0).float().sum(dim=-1).mean().item()

torch.save(
    {
        "sae_state_dict": sae.state_dict(),
        "model_name": MODEL_NAME,
        "hook_name": HOOK_NAME,
        "layer": 6,
        "d_in": D_IN,
        "d_sae": D_SAE,
        "l1_coeff": L1_COEFF,
        "epochs": SAE_EPOCHS,
        "final_mse": final_mse,
        "final_avg_l0": final_l0,
        "training_data": "python_code",
        "base_model": "code_finetuned_pythia160m"
    },
    SAVE_PATH
)

print("saved:", SAVE_PATH)
print("mse:", final_mse)
print("avg_l0:", final_l0)
