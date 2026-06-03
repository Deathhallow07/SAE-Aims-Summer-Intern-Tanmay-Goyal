import os
import torch
import pandas as pd

from common import SparseAutoencoder, D_IN, D_SAE, DEVICE, compare_decoder_directions


os.makedirs("results", exist_ok=True)

PRE_SAE_PATH = "models/pretrained_pythia160m_code_layer6_sae.pt"
FT_SAE_PATH = "models/code_finetuned_pythia160m_layer6_sae.pt"

pre_ckpt = torch.load(PRE_SAE_PATH, map_location=DEVICE)
ft_ckpt = torch.load(FT_SAE_PATH, map_location=DEVICE)

pre_sae = SparseAutoencoder(D_IN, D_SAE).to(DEVICE)
ft_sae = SparseAutoencoder(D_IN, D_SAE).to(DEVICE)

pre_sae.load_state_dict(pre_ckpt["sae_state_dict"])
ft_sae.load_state_dict(ft_ckpt["sae_state_dict"])

pre_sae.eval()
ft_sae.eval()

best_match_indices, best_match_scores = compare_decoder_directions(pre_sae, ft_sae)

comparison_df = pd.DataFrame({
    "pretrained_code_feature": list(range(D_SAE)),
    "matched_finetuned_code_feature": best_match_indices.numpy(),
    "decoder_cosine_similarity": best_match_scores.numpy()
})

def status(x):
    if x >= 0.50:
        return "relatively_preserved"
    elif x >= 0.25:
        return "partially_shifted"
    else:
        return "strongly_changed_or_unmatched"

comparison_df["status"] = comparison_df["decoder_cosine_similarity"].apply(status)

comparison_df.to_csv("results/proper_code_sae_feature_decoder_comparison.csv", index=False)

status_counts = comparison_df["status"].value_counts().reset_index()
status_counts.columns = ["status", "count"]
status_counts["percentage"] = 100 * status_counts["count"] / D_SAE

status_counts.to_csv("results/feature_status_counts.csv", index=False)

summary_df = pd.DataFrame([{
    "pre_code_mse": pre_ckpt["final_mse"],
    "pre_code_avg_l0": pre_ckpt["final_avg_l0"],
    "ft_code_mse": ft_ckpt["final_mse"],
    "ft_code_avg_l0": ft_ckpt["final_avg_l0"],
    "mean_decoder_best_cosine": best_match_scores.mean().item(),
    "median_decoder_best_cosine": best_match_scores.median().item(),
    "min_decoder_best_cosine": best_match_scores.min().item(),
    "max_decoder_best_cosine": best_match_scores.max().item(),
    "num_features": D_SAE
}])

summary_df.to_csv("results/evaluation_quantitative_summary.csv", index=False)

most_preserved = comparison_df.sort_values("decoder_cosine_similarity", ascending=False).head(30)
most_changed = comparison_df.sort_values("decoder_cosine_similarity", ascending=True).head(30)

most_preserved.to_csv("results/most_preserved_features.csv", index=False)
most_changed.to_csv("results/most_changed_features.csv", index=False)

print(summary_df)
print(status_counts)
