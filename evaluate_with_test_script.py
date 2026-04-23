
import os
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy import stats

from test_script import QNetwork
from test_script import bar_plot, test_pole_length  # reuse your functions

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
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
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

            score = test_pole_length(env, model)  # your function
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
# Stats vs baseline
# -------------------------
def cohen_d_welch(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2)) if (nx + ny - 2) > 0 else float("nan")
    d = (np.mean(y) - np.mean(x)) / s if s != 0 else float("inf")
    J = 1 - (3 / (4*(nx + ny) - 9)) if (nx + ny) > 2 else 1.0
    return d * J

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    rejections = [False]*m
    for k, (idx, p) in enumerate(indexed, start=1):
        threshold = alpha / (m - k + 1)
        if p <= threshold:
            rejections[idx] = True
        else:
            break
    return rejections

def compare_to_baseline(baseline: PolicyEvalResult, method: PolicyEvalResult, alpha: float = 0.05):
    rows = []
    assert baseline.scores.shape[0] == method.scores.shape[0], "mismatch in number of pole lengths"

    for i, L in enumerate(baseline.pole_lengths):
        xb = baseline.scores[i, :]
        xm = method.scores[i, :]

        t_stat, t_p = stats.ttest_ind(xm, xb, equal_var=False)
        mw_u, mw_p  = stats.mannwhitneyu(xm, xb, alternative="two-sided")
        d = cohen_d_welch(xb, xm)

        rows.append({
            "pole_length": L,
            "n_baseline": len(xb),
            "n_method": len(xm),
            "mean_baseline": float(np.mean(xb)),
            "mean_method": float(np.mean(xm)),
            "diff_mean": float(np.mean(xm) - np.mean(xb)),
            "t_stat": float(t_stat),
            "t_p_value": float(t_p),
            "mw_u": float(mw_u),
            "mw_p_value": float(mw_p),
            "cohen_d": float(d),
        })

    df = pd.DataFrame(rows)
    df["t_reject_holm"]  = holm_bonferroni(df["t_p_value"].tolist(), alpha=alpha)
    df["mw_reject_holm"] = holm_bonferroni(df["mw_p_value"].tolist(), alpha=alpha)

    xb_all = baseline.scores.flatten()
    xm_all = method.scores.flatten()
    t_stat_all, t_p_all = stats.ttest_ind(xm_all, xb_all, equal_var=False)
    mw_u_all, mw_p_all  = stats.mannwhitneyu(xm_all, xb_all, alternative="two-sided")
    d_all = cohen_d_welch(xb_all, xm_all)

    overall = pd.DataFrame([{
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
        "cohen_d": float(d_all),
        "t_reject_holm": np.nan,
        "mw_reject_holm": np.nan
    }])

    return df, overall

# -------------------------
# Use your bar_plot for each policy and rename the produced file
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
# Main
# -------------------------
def main():
    policies = {
    "baseline1":               "baseline_policy_1.pth",
    "strati_buffer1":          "strati_buff_policy_1.pth",
    "acl1":                    "acl_policy_1.pth",
    "exploration_diversity1":  "exploration_diversity_policy_1.pth",
    }

    pole_lengths = np.linspace(0.4, 1.8, 30)
    n_runs = 10

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

    # Save raw scores/summaries
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

    # Excel workbook
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

    # Use YOUR bar_plot: one PNG per policy
    for name, res in results.items():
        png_path = os.path.join(out_dir, f"{name}_bar_plot.png")
        save_bar_plot_with_your_fn(res, png_path)

    # Additional overlay plot (all policies)
    plt.figure(figsize=(9,5))
    for name, res in results.items():
        plt.plot(res.pole_lengths, res.means, marker="o", label=name)
    plt.xlabel("Pole length"); plt.ylabel("Average episode length")
    plt.title("CartPole performance across pole lengths")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_length_means.png"), dpi=150); plt.close()

    # Overall bar (means ± stds)
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
    plt.savefig(os.path.join(out_dir, "overall_bar.png"), dpi=150)
    plt.close()

    # Stats vs baseline
    stats_dir = os.path.join(out_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    baseline = results["baseline"]
    from scipy import stats as _stats

    def _compare(baseline, other):
        rows = []
        for i, L in enumerate(baseline.pole_lengths):
            xb, xm = baseline.scores[i, :], other.scores[i, :]
            t_stat, t_p = _stats.ttest_ind(xm, xb, equal_var=False)
            mw_u, mw_p  = _stats.mannwhitneyu(xm, xb, alternative="two-sided")
            d = cohen_d_welch(xb, xm)
            rows.append([L, len(xb), len(xm), float(np.mean(xb)), float(np.mean(xm)), float(np.mean(xm)-np.mean(xb)), float(t_stat), float(t_p), float(mw_u), float(mw_p), float(d)])
        df = pd.DataFrame(rows, columns=["pole_length","n_baseline","n_method","mean_baseline","mean_method","diff_mean","t_stat","t_p_value","mw_u","mw_p_value","cohen_d"])
        df["t_reject_holm"]  = holm_bonferroni(df["t_p_value"].tolist())
        df["mw_reject_holm"] = holm_bonferroni(df["mw_p_value"].tolist())
        xb_all, xm_all = baseline.scores.flatten(), other.scores.flatten()
        t_stat_all, t_p_all = _stats.ttest_ind(xm_all, xb_all, equal_var=False)
        mw_u_all, mw_p_all  = _stats.mannwhitneyu(xm_all, xb_all, alternative="two-sided")
        d_all = cohen_d_welch(xb_all, xm_all)
        df_all = pd.DataFrame([{"pole_length":"ALL","n_baseline":len(xb_all),"n_method":len(xm_all),"mean_baseline":float(np.mean(xb_all)),"mean_method":float(np.mean(xm_all)),"diff_mean":float(np.mean(xm_all)-np.mean(xb_all)),"t_stat":float(t_stat_all),"t_p_value":float(t_p_all),"mw_u":float(mw_u_all),"mw_p_value":float(mw_p_all),"cohen_d":float(d_all),"t_reject_holm":np.nan,"mw_reject_holm":np.nan}])
        return df, df_all

    comparisons = {}
    for name in ["strati_buffer", "acl", "exploration_diversity"]:
        df_pl, df_all = _compare(baseline, results[name])
        df_pl.to_csv(os.path.join(stats_dir, f"{name}_vs_baseline_per_length.csv"), index=False)
        df_all.to_csv(os.path.join(stats_dir, f"{name}_vs_baseline_overall.csv"), index=False)
        comparisons[name] = (df_pl, df_all)

    with pd.ExcelWriter(os.path.join(stats_dir, "stat_tests_vs_baseline.xlsx")) as writer:
        for name, (df_pl, df_all) in comparisons.items():
            df_pl.to_excel(writer, sheet_name=f"{name}_per_length", index=False)
            df_all.to_excel(writer, sheet_name=f"{name}_overall", index=False)

    print("\\n=== Statistical summary vs baseline (Holm-Bonferroni, alpha=0.05) ===")
    for name, (df_pl, df_all) in comparisons.items():
        sig_count_t  = int(df_pl["t_reject_holm"].sum())
        sig_count_mw = int(df_pl["mw_reject_holm"].sum())
        print(f"{name}: significant lengths (Welch t) = {sig_count_t}/30, (Mann-Whitney) = {sig_count_mw}/30")
        print(f"  Overall: mean diff = {float(df_all['diff_mean'].iloc[0]):.2f}, t_p = {float(df_all['t_p_value'].iloc[0]):.3g}, mw_p = {float(df_all['mw_p_value'].iloc[0]):.3g}, d = {float(df_all['cohen_d'].iloc[0]):.2f}")

if __name__ == "__main__":
    main()
