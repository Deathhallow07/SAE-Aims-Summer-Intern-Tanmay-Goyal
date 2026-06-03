import os
import torch
import pandas as pd

from common import SparseAutoencoder, D_IN, D_SAE, DEVICE, load_json, get_top_activating_tokens, show_token_context, get_feature_stats, compare_decoder_directions


os.makedirs("results", exist_ok=True)

PRE_SAE_PATH = "models/pretrained_pythia160m_code_layer6_sae.pt"
FT_SAE_PATH = "models/code_finetuned_pythia160m_layer6_sae.pt"

PRE_ACTS_PATH = "acts/pretrained_code_acts.pt"
FT_ACTS_PATH = "acts/finetuned_code_acts.pt"

PRE_TOKENS_PATH = "acts/pretrained_code_tokens.json"
FT_TOKENS_PATH = "acts/finetuned_code_tokens.json"

pre_ckpt = torch.load(PRE_SAE_PATH, map_location=DEVICE)
ft_ckpt = torch.load(FT_SAE_PATH, map_location=DEVICE)

pre_sae = SparseAutoencoder(D_IN, D_SAE).to(DEVICE)
ft_sae = SparseAutoencoder(D_IN, D_SAE).to(DEVICE)

pre_sae.load_state_dict(pre_ckpt["sae_state_dict"])
ft_sae.load_state_dict(ft_ckpt["sae_state_dict"])

pre_sae.eval()
ft_sae.eval()

pre_acts = torch.load(PRE_ACTS_PATH)
ft_acts = torch.load(FT_ACTS_PATH)

pre_tokens = load_json(PRE_TOKENS_PATH)
ft_tokens = load_json(FT_TOKENS_PATH)

ft_stats = get_feature_stats(ft_sae, ft_acts)

ft_to_pre_indices, ft_to_pre_scores = compare_decoder_directions(ft_sae, pre_sae)

ft_new_df = pd.DataFrame({
    "ft_feature": list(range(D_SAE)),
    "matched_pre_code_feature": ft_to_pre_indices.numpy(),
    "decoder_cosine_similarity_to_pre_code": ft_to_pre_scores.numpy(),
    "firing_rate": ft_stats["firing_rate"].numpy(),
    "max_activation": ft_stats["max_activation"].numpy()
})

candidate_new_df = ft_new_df[
    (ft_new_df["decoder_cosine_similarity_to_pre_code"] < 0.30) &
    (ft_new_df["firing_rate"] > 0.002) &
    (ft_new_df["firing_rate"] < 0.10) &
    (ft_new_df["max_activation"] > 0.5)
].sort_values("decoder_cosine_similarity_to_pre_code")

candidate_new_df.to_csv("results/candidate_new_code_features_strict.csv", index=False)

chosen_features = candidate_new_df.head(20)["ft_feature"].tolist()

rows = []

for feat in chosen_features:
    top_rows = get_top_activating_tokens(
        sae=ft_sae,
        activations=ft_acts,
        token_strs=ft_tokens,
        feature_idx=feat,
        top_k=8
    )

    for row in top_rows:
        rows.append({
            "ft_feature": feat,
            "similarity_to_pretrained": ft_to_pre_scores[feat].item(),
            "firing_rate": ft_stats["firing_rate"][feat].item(),
            "max_activation": ft_stats["max_activation"][feat].item(),
            "activation": row["activation"],
            "token": row["token"],
            "context": show_token_context(ft_tokens, row["index"], window=12)
        })

context_df = pd.DataFrame(rows)
context_df.to_csv("results/qualitative_context_examples.csv", index=False)

print(candidate_new_df.head(20))
print(context_df.head(20))
