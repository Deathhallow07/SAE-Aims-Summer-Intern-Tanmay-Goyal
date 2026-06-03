import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


MODEL_NAME = "EleutherAI/pythia-160m"
HOOK_NAME = "blocks.6.hook_resid_post"

MAX_LENGTH = 128
MAX_ACTIVATIONS = 50000

D_IN = 768
D_SAE = 4096

SAE_BATCH_SIZE = 512
SAE_LR = 1e-3
L1_COEFF = 1e-4
SAE_EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_sae):
        super().__init__()

        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def encode(self, x):
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, features):
        return features @ self.W_dec + self.b_dec

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-6)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


@torch.no_grad()
def get_top_activating_tokens(sae, activations, token_strs, feature_idx, top_k=20, batch_size=1024):
    sae.eval()

    all_scores = []

    for start in range(0, activations.shape[0], batch_size):
        end = start + batch_size
        x = activations[start:end].to(DEVICE)

        features = sae.encode(x)
        scores = features[:, feature_idx].detach().cpu()
        all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0)
    top_scores, top_indices = torch.topk(all_scores, k=top_k)

    results = []

    for score, idx in zip(top_scores, top_indices):
        idx = idx.item()
        results.append({
            "activation": score.item(),
            "token": token_strs[idx],
            "index": idx
        })

    return results


def show_token_context(token_strs, center_idx, window=12):
    start = max(0, center_idx - window)
    end = min(len(token_strs), center_idx + window + 1)

    context = token_strs[start:end].copy()
    center_local_idx = center_idx - start
    context[center_local_idx] = f"[[{context[center_local_idx]}]]"

    return "".join(context)


@torch.no_grad()
def get_feature_stats(sae, activations, batch_size=1024):
    sae.eval()

    feature_sums = torch.zeros(sae.d_sae)
    feature_counts = torch.zeros(sae.d_sae)
    feature_max = torch.zeros(sae.d_sae)

    for start in tqdm(range(0, activations.shape[0], batch_size)):
        end = start + batch_size
        x = activations[start:end].to(DEVICE)

        features = sae.encode(x).detach().cpu()

        feature_sums += features.sum(dim=0)
        feature_counts += (features > 0).float().sum(dim=0)
        feature_max = torch.maximum(feature_max, features.max(dim=0).values)

    firing_rate = feature_counts / activations.shape[0]
    mean_activation = feature_sums / activations.shape[0]

    return {
        "firing_rate": firing_rate,
        "mean_activation": mean_activation,
        "max_activation": feature_max
    }


@torch.no_grad()
def compare_decoder_directions(sae_a, sae_b, batch_size=512):
    W_a = sae_a.W_dec.detach().cpu()
    W_b = sae_b.W_dec.detach().cpu()

    W_a = F.normalize(W_a, dim=1)
    W_b = F.normalize(W_b, dim=1)

    all_best_scores = []
    all_best_indices = []

    for start in tqdm(range(0, W_a.shape[0], batch_size)):
        end = start + batch_size

        sims = W_a[start:end] @ W_b.T
        best_scores, best_indices = sims.max(dim=1)

        all_best_scores.append(best_scores)
        all_best_indices.append(best_indices)

    best_match_scores = torch.cat(all_best_scores)
    best_match_indices = torch.cat(all_best_indices)

    return best_match_indices, best_match_scores
