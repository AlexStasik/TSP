# %%
# ============================================================
# THRESHOLD GROVER FOR TSP (small-n simulator prototype)
#
# Idea:
#   1) Prepare uniform superposition over time-register bitstrings.
#   2) Mark tours with cost_norm <= tau by an exact lookup oracle.
#   3) Apply Grover iterations on the time register.
#   4) Sweep multiple tau values and report best-path concentration.
#
# Notes:
# - Uses lookup over all valid tours, so this is not scalable.
# - Designed for n=5 style experiments and quick hypothesis testing.
# ============================================================

import math
import itertools
import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf

# -------------------------
# Problem setup
# -------------------------
np.random.seed(42)

n = 5
start_node = n - 1
shots = 5000

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)

# Cost normalization shared with other scripts.
C_max = float(np.max(all_costs_raw))
cost_matrix = cost_matrix_raw / C_max

# -------------------------
# Encoding
# -------------------------
T_steps = n - 1
n_qubits_step = int(np.ceil(np.log2(n - 1)))


def twire(t, q):
    return f"t{t}_q{q}"


time_wires = [twire(t, q) for t in range(T_steps) for q in range(n_qubits_step)]
mark_anc = "mark"
diff_anc = "diff"
wires = time_wires + [mark_anc, diff_anc]

m = len(time_wires)
N = 2**m

dev_samp = qml.device("lightning.qubit", wires=wires, shots=shots)

# -------------------------
# Helpers
# -------------------------


def int_to_bits(x: int, width: int):
    return [(x >> (width - 1 - k)) & 1 for k in range(width)]


def apply_controls_for_value(wires_list, value_bits):
    for w, b in zip(wires_list, value_bits):
        if int(b) == 0:
            qml.PauliX(wires=w)


def undo_controls_for_value(wires_list, value_bits):
    for w, b in zip(wires_list, value_bits):
        if int(b) == 0:
            qml.PauliX(wires=w)


def decode_path_from_time_bits(sample_row):
    bits = "".join(str(int(b)) for b in sample_row)
    path = []
    for t in range(T_steps):
        chunk = bits[t * n_qubits_step : (t + 1) * n_qubits_step]
        path.append(int(chunk, 2))
    return path


def is_valid_classical(path):
    return sorted(path) == list(range(n - 1))


def tour_cost_norm_from_indices(path_idx):
    c = float(cost_matrix[start_node, path_idx[0]])
    for t in range(T_steps - 1):
        c += float(cost_matrix[path_idx[t], path_idx[t + 1]])
    c += float(cost_matrix[path_idx[-1], start_node])
    return float(c)


def grover_iterations_from_M(M_marked: int):
    if M_marked <= 0 or M_marked >= N:
        return 0
    theta = math.asin(math.sqrt(M_marked / N))
    return int(math.floor((math.pi / (4.0 * theta)) - 0.5)) if theta > 0 else 0


# -------------------------
# Classical valid-tour table
# -------------------------

valid_tours = list(itertools.permutations(range(n - 1)))
valid_costs_norm = np.array([tour_cost_norm_from_indices(t) for t in valid_tours], dtype=float)

tour_bitstrings = []
for tour in valid_tours:
    bits = []
    for t in range(T_steps):
        bits.extend(int_to_bits(int(tour[t]), n_qubits_step))
    tour_bitstrings.append(bits)
tour_bitstrings = np.array(tour_bitstrings, dtype=int)

best_idx = int(np.argmin(valid_costs_norm))
best_path = list(valid_tours[best_idx])
best_path_str = str(best_path)
best_cost = float(valid_costs_norm[best_idx])

# Build threshold schedule by target marked counts among valid tours.
target_marked_counts = [24, 16, 12, 8, 6, 4, 3, 2, 1]
sorted_unique_costs = np.unique(np.sort(valid_costs_norm))
thresholds = []
for target in target_marked_counts:
    idx = int(min(max(target - 1, 0), len(sorted_unique_costs) - 1))
    thresholds.append(float(sorted_unique_costs[idx]))

# preserve order, remove duplicates
seen = set()
thresholds = [x for x in thresholds if not (x in seen or seen.add(x))]


# -------------------------
# Quantum blocks
# -------------------------


def init_uniform():
    for w in time_wires:
        qml.Hadamard(wires=w)


def phase_oracle_threshold(marked_indices):
    for i in marked_indices:
        bits = tour_bitstrings[i].tolist()
        apply_controls_for_value(time_wires, bits)

        qml.MultiControlledX(wires=time_wires + [mark_anc])
        qml.PauliZ(wires=mark_anc)
        qml.MultiControlledX(wires=time_wires + [mark_anc])

        undo_controls_for_value(time_wires, bits)


def diffusion_time_register():
    for w in time_wires:
        qml.Hadamard(wires=w)
    for w in time_wires:
        qml.PauliX(wires=w)

    qml.MultiControlledX(wires=time_wires + [diff_anc])
    qml.PauliZ(wires=diff_anc)
    qml.MultiControlledX(wires=time_wires + [diff_anc])

    for w in time_wires:
        qml.PauliX(wires=w)
    for w in time_wires:
        qml.Hadamard(wires=w)


def grover_step(marked_indices):
    phase_oracle_threshold(marked_indices)
    diffusion_time_register()


# -------------------------
# Sweep
# -------------------------


def run_for_threshold(tau: float, tau_id: int, mode: str = "grid", prefix: str = "tau"):
    marked_indices = [i for i, c in enumerate(valid_costs_norm) if c <= (tau + 1e-12)]
    M_marked = int(len(marked_indices))
    K = int(grover_iterations_from_M(M_marked))

    if M_marked > 0:
        theta = math.asin(math.sqrt(M_marked / N))
        p_marked_after = float((math.sin((2 * K + 1) * theta)) ** 2)
    else:
        p_marked_after = 0.0

    print("\n========================================")
    print(f"{mode} {prefix}[{tau_id}] =", float(tau))
    print("Marked valid tours M =", M_marked, "out of", len(valid_tours), "(N time states =", N, ")")
    print("Chosen Grover iterations K =", K)
    print("Predicted marked-state probability after K:", p_marked_after)

    @qml.qnode(dev_samp)
    def sample_circuit():
        init_uniform()
        for _ in range(K):
            grover_step(marked_indices)
        return qml.sample(wires=time_wires)

    samples = sample_circuit()

    rows = []
    for r in range(shots):
        path = decode_path_from_time_bits(samples[r])
        valid = int(is_valid_classical(path))
        if valid:
            c_norm = float(tour_cost_norm_from_indices(path))
            under_tau = int(c_norm <= (tau + 1e-12))
        else:
            c_norm = np.nan
            under_tau = 0

        rows.append(
            {
                "path": path,
                "valid": valid,
                "cost_norm": c_norm,
                "is_under_tau": under_tau,
                "tau": float(tau),
                "K_iterations": K,
                "M_marked_valid": M_marked,
            }
        )

    df = pd.DataFrame(rows)
    valid_rate = float(df["valid"].mean())
    marked_rate_all = float(df["is_under_tau"].mean())

    valid_df = df[df["valid"] == 1].copy()
    if len(valid_df) > 0:
        marked_rate_given_valid = float(valid_df["is_under_tau"].mean())
        mean_valid_cost = float(valid_df["cost_norm"].mean())
        valid_df["path_str"] = valid_df["path"].astype(str)
        ranking = (
            valid_df.groupby("path_str", as_index=False)
            .agg(
                count=("path_str", "size"),
                cost_norm=("cost_norm", "mean"),
                under_tau_rate=("is_under_tau", "mean"),
            )
            .sort_values(["count", "cost_norm"], ascending=[False, True])
            .reset_index(drop=True)
        )
        ranking["measured_rank"] = ranking["count"].rank(
            ascending=False, method="dense"
        ).astype(int)
        ranking["classical_rank"] = ranking["cost_norm"].rank(
            ascending=True, method="dense"
        ).astype(int)
    else:
        marked_rate_given_valid = np.nan
        mean_valid_cost = np.nan
        ranking = pd.DataFrame(
            columns=["path_str", "count", "cost_norm", "under_tau_rate", "measured_rank", "classical_rank"]
        )

    best_row = ranking[ranking["path_str"] == best_path_str]
    if len(best_row) > 0:
        best_count = int(best_row.iloc[0]["count"])
        best_rank = int(best_row.iloc[0]["measured_rank"])
    else:
        best_count = 0
        best_rank = np.nan

    best_p_all = float(best_count / shots)
    valid_count = int(df["valid"].sum())
    best_p_given_valid = float(best_count / valid_count) if valid_count > 0 else np.nan

    print("Measured valid fraction:", valid_rate)
    print("Measured under-tau fraction (all shots):", marked_rate_all)
    print("Best-path probability (all shots):", best_p_all)
    print("Best-path measured rank:", best_rank)

    samples_out = f"tsp_threshold_{prefix}{tau_id:02d}_samples.csv"
    ranking_out = f"tsp_threshold_{prefix}{tau_id:02d}_ranking.csv"
    df.to_csv(samples_out, index=False)
    ranking.to_csv(ranking_out, index=False)
    print("Saved:", samples_out)
    print("Saved:", ranking_out)

    return {
        "tau_id": tau_id,
        "mode": mode,
        "tau": float(tau),
        "M_marked_valid": M_marked,
        "K_iterations": K,
        "p_marked_after_predicted": p_marked_after,
        "valid_rate": valid_rate,
        "under_tau_rate_all_shots": marked_rate_all,
        "under_tau_rate_given_valid": marked_rate_given_valid,
        "best_path": best_path_str,
        "best_path_cost_norm": best_cost,
        "best_path_count": best_count,
        "best_path_p_all_shots": best_p_all,
        "best_path_p_given_valid": best_p_given_valid,
        "best_path_measured_rank": best_rank,
        "mean_valid_cost_norm": mean_valid_cost,
        "samples_csv": samples_out,
        "ranking_csv": ranking_out,
    }


def main():
    print("Best valid tour by classical cost:", best_path, "cost_norm:", best_cost)
    print("Threshold sweep schedule:", thresholds)

    summary_rows = []
    for i, tau in enumerate(thresholds, start=1):
        summary_rows.append(run_for_threshold(tau=tau, tau_id=i, mode="grid", prefix="tau"))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["best_path_p_all_shots", "valid_rate"], ascending=[False, False]
    ).reset_index(drop=True)

    summary_out = "tsp_threshold_grover_sweep_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print("\nSaved:", summary_out)
    print("\nSweep summary (top-10 by best-path probability):")
    print(
        summary_df[
            [
                "tau_id",
                "tau",
                "M_marked_valid",
                "K_iterations",
                "best_path_p_all_shots",
                "best_path_measured_rank",
                "valid_rate",
                "under_tau_rate_all_shots",
                "mean_valid_cost_norm",
            ]
        ].head(10).to_string(index=False)
    )

    # Adaptive threshold loop: tighten tau from sampled valid-cost quantiles.
    adaptive_rounds = 6
    adaptive_quantile = 0.25
    tau_current = float(np.max(valid_costs_norm))
    adaptive_rows = []

    print("\n========================================")
    print("Starting adaptive threshold loop")
    print("rounds =", adaptive_rounds, "quantile =", adaptive_quantile)

    for r in range(1, adaptive_rounds + 1):
        row = run_for_threshold(
            tau=tau_current,
            tau_id=r,
            mode="adaptive",
            prefix="adapt_round",
        )
        adaptive_rows.append(row)

        # Stop early once threshold isolates a single marked tour.
        if int(row["M_marked_valid"]) <= 1:
            break

        # Use sampled valid costs to tighten tau adaptively.
        sample_df = pd.read_csv(row["samples_csv"])
        valid_cost_samples = sample_df.loc[sample_df["valid"] == 1, "cost_norm"].dropna().to_numpy()
        if len(valid_cost_samples) == 0:
            break

        tau_candidate = float(np.quantile(valid_cost_samples, adaptive_quantile))

        # Snap to achievable tour-cost levels and force strict tightening.
        leq = sorted_unique_costs[sorted_unique_costs <= (tau_candidate + 1e-12)]
        if len(leq) == 0:
            break
        tau_next = float(leq[-1])
        if tau_next >= (tau_current - 1e-12):
            lower = sorted_unique_costs[sorted_unique_costs < (tau_current - 1e-12)]
            if len(lower) == 0:
                break
            tau_next = float(lower[-1])

        tau_current = tau_next

    adaptive_df = pd.DataFrame(adaptive_rows).sort_values(
        ["best_path_p_all_shots", "valid_rate"], ascending=[False, False]
    ).reset_index(drop=True)
    adaptive_out = "tsp_threshold_grover_adaptive_summary.csv"
    adaptive_df.to_csv(adaptive_out, index=False)
    print("\nSaved:", adaptive_out)
    if len(adaptive_df) > 0:
        print("\nAdaptive summary:")
        print(
            adaptive_df[
                [
                    "tau_id",
                    "tau",
                    "M_marked_valid",
                    "K_iterations",
                    "best_path_p_all_shots",
                    "best_path_measured_rank",
                    "valid_rate",
                    "under_tau_rate_all_shots",
                    "mean_valid_cost_norm",
                ]
            ].to_string(index=False)
        )

    # Export overall best configuration (grid + adaptive) for quick inspection.
    all_df = pd.concat([summary_df, adaptive_df], ignore_index=True)
    all_df = all_df.sort_values(
        ["best_path_p_all_shots", "valid_rate"], ascending=[False, False]
    ).reset_index(drop=True)
    all_out = "tsp_threshold_grover_all_runs_summary.csv"
    all_df.to_csv(all_out, index=False)
    print("Saved:", all_out)

    best = all_df.iloc[0]
    best_samples = str(best["samples_csv"])
    best_ranking = str(best["ranking_csv"])
    pd.read_csv(best_samples).to_csv("tsp_threshold_grover_best_samples.csv", index=False)
    pd.read_csv(best_ranking).to_csv("tsp_threshold_grover_best_ranking.csv", index=False)
    print("Saved:", "tsp_threshold_grover_best_samples.csv")
    print("Saved:", "tsp_threshold_grover_best_ranking.csv")


if __name__ == "__main__":
    main()
# %%
