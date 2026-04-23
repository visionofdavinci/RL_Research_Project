import os
import math
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy import stats

# Reuse your module & helpers
import test_script as ts
from test_script import QNetwork, bar_plot, test_pole_length, test_script

# -------------------------
# Evaluation utilities
# -------------------------
@dataclass
class PolicyEvalResult:
    name: str
    weights_path: str
    pole_lengths: List[float]
    n_runs: int
    scores: np.ndarray  # shape (num_lengths, n_runs)
    means: np.ndarray   # shape (num_lengths, )
    stds: np.ndarray    # shape (num_lengths, )
    overall_mean: float
    overall_std: float

def load_model(weights_path: str, env_name: str = "CartPole-v1") -> Tuple[QNetwork, int, int]:
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = QNetwork(state_dim, action_dim)
    # Load weights
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions without weights_only parameter
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    env.close()
    return model, state_dim, action_dim

def evaluate_policy(weights_path: str, name: str, pole_lengths: np.ndarray, n_runs: int = 10, env_name: str = "CartPole-v1") -> PolicyEvalResult:
    model, _, _ = load_model(weights_path, env_name=env_name)

    scores = np.zeros((len(pole_lengths), n_runs), dtype=float)

    for i, L in enumerate(pole_lengths):
        for r in range(n_runs):
            env = gym.make(env_name)
            env.unwrapped.length = float(L)
            score = test_pole_length(env, model)  # your fixed eval
            scores[i, r] = score
            env.close()

    means = scores.mean(axis=1)
    stds = scores.std(axis=1, ddof=1)
    overall_mean = scores.mean()
    overall_std  = scores.std(ddof=1)

    return PolicyEvalResult(
        name=name,
        weights_path=weights_path,
        pole_lengths=list(map(float, pole_lengths)),
        n_runs=n_runs,
        scores=scores,
        means=means,
        stds=stds,
        overall_mean=float(overall_mean),
        overall_std=float(overall_std),
    )

# -------------------------
# Stats helpers
# -------------------------
def hedges_g(x, y):
    """
    Calculate Hedges' g effect size (small-sample corrected Cohen's d).
    
    Hedges' g = J * Cohen's d, where J is a correction factor.
    This is more appropriate for small samples (n < 20 per group).
    """
    nx, ny = len(x), len(y)
    
    if nx < 2 or ny < 2:
        return float("nan")
    
    # Calculate pooled standard deviation
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    
    if pooled_sd == 0:
        return float("inf") if np.mean(y) != np.mean(x) else 0.0
    
    # Cohen's d
    d = (np.mean(y) - np.mean(x)) / pooled_sd
    
    # Small-sample correction factor (Hedges' correction)
    df = nx + ny - 2
    if df > 0:
        J = 1 - (3 / (4 * df - 1))
    else:
        J = 1.0
    
    # Hedges' g
    g = d * J
    
    return g

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Returns a list of booleans indicating whether each null hypothesis is rejected.
    """
    m = len(p_values)
    if m == 0:
        return []
    
    # Create list of (index, p_value) and sort by p_value
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    rejections = [False] * m
    
    # Sequential testing with adjusted thresholds
    for k, (idx, p) in enumerate(indexed, start=1):
        threshold = alpha / (m - k + 1)
        if p <= threshold:
            rejections[idx] = True
        else:
            # Once we fail to reject, stop (all subsequent tests also fail)
            break
    
    return rejections

def compare_to_baseline(baseline: PolicyEvalResult, method: PolicyEvalResult, alpha: float = 0.05):
    """
    Compare a method to baseline using Welch's t-test and Mann-Whitney U test per pole length.
    Apply Holm-Bonferroni correction across the 30 tests.
    """
    rows = []
    assert baseline.scores.shape[0] == method.scores.shape[0], "mismatch in number of pole lengths"

    for i, L in enumerate(baseline.pole_lengths):
        xb = baseline.scores[i, :]
        xm = method.scores[i, :]
        
        # Welch's t-test (unpaired, unequal variances)
        t_stat, t_p = stats.ttest_ind(xm, xb, equal_var=False)
        
        # Mann-Whitney U test (non-parametric alternative)
        mw_u, mw_p = stats.mannwhitneyu(xm, xb, alternative="two-sided")
        
        # Hedges' g effect size
        g = hedges_g(xb, xm)
        
        rows.append([
            L, 
            len(xb), 
            len(xm), 
            float(np.mean(xb)), 
            float(np.mean(xm)), 
            float(np.mean(xm) - np.mean(xb)), 
            float(t_stat), 
            float(t_p), 
            float(mw_u), 
            float(mw_p), 
            float(g)
        ])

    df = pd.DataFrame(rows, columns=[
        "pole_length", "n_baseline", "n_method", 
        "mean_baseline", "mean_method", "diff_mean", 
        "t_stat", "t_p_value", "mw_u", "mw_p_value", "hedges_g"
    ])
    
    # Apply Holm-Bonferroni correction across all 30 tests
    df["t_reject_holm"] = holm_bonferroni(df["t_p_value"].tolist(), alpha=alpha)
    df["mw_reject_holm"] = holm_bonferroni(df["mw_p_value"].tolist(), alpha=alpha)

    # Overall aggregate test (all lengths and runs flattened)
    xb_all = baseline.scores.flatten()
    xm_all = method.scores.flatten()
    
    t_stat_all, t_p_all = stats.ttest_ind(xm_all, xb_all, equal_var=False)
    mw_u_all, mw_p_all = stats.mannwhitneyu(xm_all, xb_all, alternative="two-sided")
    g_all = hedges_g(xb_all, xm_all)
    
    df_all = pd.DataFrame([{
        "pole_length": "ALL",
        "n_baseline": len(xb_all),
        "n_method": len(xm_all),
        "mean_baseline": float(np.mean(xb_all)),
        "mean_method": float(np.mean(xm_all)),
        "diff_mean": float(np.mean(xm_all) - np.mean(xb_all)),
        "t_stat": float(t_stat_all),
        "t_p_value": float(t_p_all),
        "mw_u": float(mw_u_all),
        "mw_p_value": float(mw_p_all),
        "hedges_g": float(g_all),
        "t_reject_holm": np.nan,
        "mw_reject_holm": np.nan
    }])
    
    return df, df_all

# -------------------------
# Use your bar_plot per policy
# -------------------------
def save_bar_plot_with_your_fn(res: PolicyEvalResult, out_path: str):
    data = {}
    total = 0.0
    for L, mean, std in zip(res.pole_lengths, res.means, res.stds):
        key_avg = f"Avg_{round(L, 2)}"
        key_std = f"Std_{round(L, 2)}"
        data[key_avg] = float(mean)
        data[key_std] = float(std)
        total += float(mean)
    data["Total"] = float(total)

    bar_plot([data])  # writes bar_plot.png
    default_png = "bar_plot.png"
    if os.path.exists(default_png):
        os.replace(default_png, out_path)

# -------------------------
# Integrate and run ts.test_script() per policy
# -------------------------
def run_legacy_test_script_for_policy(weights_path: str, policy_name: str, output_dir: str):
    """
    Run the original test_script() function for a policy.
    Directly passes the weights path to test_script() and moves outputs.
    """
    print(f"  Running test_script() with {weights_path}")
    
    # Run test_script() with the actual weights path
    ts.test_script(weights_path=weights_path)

    # Move outputs to output directory with policy name
    src_png = "bar_plot.png"
    src_xlsx = "experiment_results.xlsx"
    
    if os.path.exists(src_png):
        dst_png = os.path.join(output_dir, f"{policy_name}_legacy_bar_plot.png")
        os.replace(src_png, dst_png)
    else:
        print(f"  Warning: {src_png} not found")
        
    if os.path.exists(src_xlsx):
        dst_xlsx = os.path.join(output_dir, f"{policy_name}_legacy_experiment_results.xlsx")
        os.replace(src_xlsx, dst_xlsx)
    else:
        print(f"  Warning: {src_xlsx} not found")

# -------------------------
# Main
# -------------------------
def main():
    # Direct filenames in the current directory first; fallback to ./weights/
    policies = {
        "baseline_optimized":               "baseline_optimized_policy.pth",
        "strati_buffer_optimized":          "strati_buff_optimized_policy.pth",
        "acl_optimized":                    "acl_optimized_policy.pth",
        "exploration_diversity_optimized":  "exploration_diversity_optimized_policy.pth",
    }
    
    # Check for files in current directory first, then weights/
    for k, v in list(policies.items()):
        if not os.path.isfile(v):
            alt = os.path.join("weights", v)
            if os.path.isfile(alt):
                policies[k] = alt

    missing = [n for n, p in policies.items() if not os.path.isfile(p)]
    if missing:
        print("Missing model files:", ", ".join(missing))
        print("Place them in the current folder (recommended) or provide paths in the script.")
        return

    pole_lengths = np.linspace(0.4, 1.8, 30)
    n_runs = 10

    # -------------------------
    # Evaluate all policies
    # -------------------------
    results: Dict[str, PolicyEvalResult] = {}
    t0 = time.time()
    for name, path in policies.items():
        print(f"Evaluating {name} from {path} ...")
        res = evaluate_policy(path, name, pole_lengths=pole_lengths, n_runs=n_runs, env_name="CartPole-v1")
        results[name] = res
        print(f"  -> overall mean={res.overall_mean:.2f} std={res.overall_std:.2f}")
    print(f"Evaluation finished in {time.time()-t0:.1f}s")

    out_dir = os.path.join(os.getcwd(), "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    # Create images subdirectory for all plots
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # -------------------------
    # Save raw scores and summaries
    # -------------------------
    for name, res in results.items():
        df_scores = pd.DataFrame(res.scores, columns=[f"run_{i+1}" for i in range(res.n_runs)])
        df_scores.insert(0, "pole_length", res.pole_lengths)
        df_scores.to_csv(os.path.join(out_dir, f"{name}_scores.csv"), index=False)

        df_summary = pd.DataFrame({
            "pole_length": res.pole_lengths,
            "mean": res.means,
            "std": res.stds,
        })
        df_summary.to_csv(os.path.join(out_dir, f"{name}_summary.csv"), index=False)

    # -------------------------
    # Excel workbook with all results
    # -------------------------
    with pd.ExcelWriter(os.path.join(out_dir, "summaries.xlsx")) as writer:
        for name, res in results.items():
            df_scores = pd.DataFrame(res.scores, columns=[f"run_{i+1}" for i in range(res.n_runs)])
            df_scores.insert(0, "pole_length", res.pole_lengths)
            df_scores.to_excel(writer, sheet_name=f"{name}_scores", index=False)

            df_summary = pd.DataFrame({
                "pole_length": res.pole_lengths,
                "mean": res.means,
                "std": res.stds,
            })
            df_summary.to_excel(writer, sheet_name=f"{name}_summary", index=False)

        overall_df = pd.DataFrame({
            "policy": list(results.keys()),
            "overall_mean": [results[k].overall_mean for k in results.keys()],
            "overall_std":  [results[k].overall_std  for k in results.keys()],
        })
        overall_df.to_excel(writer, sheet_name="overall", index=False)

    # -------------------------
    # Use your bar_plot: one PNG per policy
    # -------------------------
    for name, res in results.items():
        png_path = os.path.join(images_dir, f"{name}_bar_plot.png")
        save_bar_plot_with_your_fn(res, png_path)

    # -------------------------
    # ALSO run your original test_script() once per policy
    # -------------------------
    for name, path in policies.items():
        print(f"Running original test_script() for {name} ...")
        run_legacy_test_script_for_policy(path, name, images_dir)

    # -------------------------
    # Extra overlay plot for comparison
    # -------------------------
    plt.figure(figsize=(9,5))
    for name, res in results.items():
        plt.plot(res.pole_lengths, res.means, marker="o", label=name)
    plt.xlabel("Pole length")
    plt.ylabel("Average episode length")
    plt.title("CartPole performance across pole lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "per_length_means.png"), dpi=150)
    plt.close()

    # -------------------------
    # Overall bar (means ± stds)
    # -------------------------
    names = list(results.keys())
    means = [results[n].overall_mean for n in names]
    stds  = [results[n].overall_std for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(8,5))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    plt.xticks(x, names, rotation=15)
    plt.ylabel("Average episode length (overall)")
    plt.title("Overall performance")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "overall_bar.png"), dpi=150)
    plt.close()

    # -------------------------
    # Statistical tests vs baseline
    # -------------------------
    stats_dir = os.path.join(out_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    baseline = results["baseline_optimized"]
    comparisons = {}
    for name in ["strati_buffer_optimized", "acl_optimized", "exploration_diversity_optimized"]:
        df_pl, df_all = compare_to_baseline(baseline, results[name], alpha=0.05)
        df_pl.to_csv(os.path.join(stats_dir, f"{name}_vs_baseline_per_length.csv"), index=False)
        df_all.to_csv(os.path.join(stats_dir, f"{name}_vs_baseline_overall.csv"), index=False)
        comparisons[name] = (df_pl, df_all)

    with pd.ExcelWriter(os.path.join(stats_dir, "stat_tests_vs_baseline.xlsx")) as writer:
        for name, (df_pl, df_all) in comparisons.items():
            df_pl.to_excel(writer, sheet_name=f"{name}_per_length", index=False)
            df_all.to_excel(writer, sheet_name=f"{name}_overall", index=False)

    # -------------------------
    # Print statistical summary
    # -------------------------
    print("\n=== Statistical summary vs baseline (Holm-Bonferroni, alpha=0.05) ===")
    for name, (df_pl, df_all) in comparisons.items():
        sig_count_t  = int(df_pl["t_reject_holm"].sum())
        sig_count_mw = int(df_pl["mw_reject_holm"].sum())
        print(f"\n{name}:")
        print(f"  Significant lengths (Welch t-test): {sig_count_t}/30")
        print(f"  Significant lengths (Mann-Whitney): {sig_count_mw}/30")
        print(f"  Overall: mean diff = {float(df_all['diff_mean'].iloc[0]):.2f}, " +
              f"t_p = {float(df_all['t_p_value'].iloc[0]):.3g}, " +
              f"mw_p = {float(df_all['mw_p_value'].iloc[0]):.3g}, " +
              f"Hedges' g = {float(df_all['hedges_g'].iloc[0]):.3f}")

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"All images saved to: {images_dir}")
    print(f"Statistical tests saved to: {stats_dir}")

if __name__ == "__main__":
    main()