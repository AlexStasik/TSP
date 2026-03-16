# %%
# ============================================================
# OPTION 1 IMPLEMENTATION:
#   - Nonnegative quality map: x = 1 - scaled_cost in [0, 1]
#   - Monotonic QSVT filter: P(x) = x^k
#
# Pipeline:
#   Prep (uniform time register)
#   -> Validity compute (good wire)
#   -> QSVT on UA (signal wire)
#   -> AA on Success = (good=1 AND sig=0)
#   -> sample time register and report ranking
#
# Notes:
# - Small-n simulator experiment (not scalable).
# - Uses the same TSP encoding style as the baseline script.
# ============================================================

import math
import itertools
import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf

# -------------------------
# Problem + normalized costs
# -------------------------
np.random.seed(42)

n = 5
start_node = n - 1

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)

# Normalize once for phase-friendly values.
C_max = float(np.max(all_costs_raw))
cost_matrix = cost_matrix_raw / C_max

# -------------------------
# Encoding
# -------------------------
T_steps = n - 1
n_qubits_step = int(np.ceil(np.log2(n - 1)))


def twire(t, q):
    return f"t{t}_q{q}"


def pos_wires_at_t(t):
    return [twire(t, q) for q in range(n_qubits_step)]


time_wires = [twire(t, q) for t in range(T_steps) for q in range(n_qubits_step)]
state_wires = [f"s{i}" for i in range(n - 1)]

good_wire = "good"
sig_wire = "sig"
diff_anc = "diff"
flag_anc = "flag"

m = len(time_wires)
# Keep the active register set minimal to avoid unnecessary simulator memory pressure.
wires = time_wires + state_wires + [good_wire, sig_wire, diff_anc, flag_anc]
# A acts on these wires; S0 should reflect this subspace only.
a_wires = time_wires + state_wires + [good_wire, sig_wire]

dev_ana = qml.device("lightning.qubit", wires=wires)
shots = 5000
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


# -------------------------
# Validity compute
# -------------------------


def init_pos_uniform():
    for w in time_wires:
        qml.Hadamard(wires=w)


def check_pos_compute():
    for t in range(T_steps):
        pos = pos_wires_at_t(t)
        for i in range(n - 1):
            bits = format(i, f"0{n_qubits_step}b")
            for w, b in zip(pos, bits):
                if b == "0":
                    qml.PauliX(wires=w)

            qml.MultiControlledX(wires=pos + [state_wires[i]])

            for w, b in zip(pos, bits):
                if b == "0":
                    qml.PauliX(wires=w)

    qml.MultiControlledX(wires=state_wires + [good_wire])


# -------------------------
# UA lookup over valid tours
# -------------------------


def all_valid_tours_indices(n_):
    for perm in itertools.permutations(range(n_ - 1)):
        yield perm


valid_tours = list(all_valid_tours_indices(n))
valid_costs_norm = np.array(
    [tour_cost_norm_from_indices(path) for path in valid_tours], dtype=float
)

# Option 1: map cost to x in [0, 1], where best tour has x=1.
cost_min = float(np.min(valid_costs_norm))
cost_max = float(np.max(valid_costs_norm))
cost_span = float(max(cost_max - cost_min, 1e-12))
costs_scaled = (valid_costs_norm - cost_min) / cost_span
x_vals = 1.0 - costs_scaled
thetas = np.arccos(np.clip(x_vals, -1.0, 1.0))

tour_bitstrings = []
for tour in valid_tours:
    bits = []
    for t in range(T_steps):
        bits.extend(int_to_bits(int(tour[t]), n_qubits_step))
    tour_bitstrings.append(bits)
tour_bitstrings = np.array(tour_bitstrings, dtype=int)


def apply_UA():
    for bits, theta in zip(tour_bitstrings, thetas):
        apply_controls_for_value(time_wires, bits.tolist())

        qml.MultiControlledX(wires=time_wires + [sig_wire])
        qml.RY(2.0 * float(theta), wires=sig_wire)
        qml.MultiControlledX(wires=time_wires + [sig_wire])

        undo_controls_for_value(time_wires, bits.tolist())


# -------------------------
# QSVT filter
# -------------------------


def apply_qsvt_from_angles(angles, apply_ua_fn, work_wire):
    def pc(phi):
        qml.PCPhase(phi, dim=1, wires=work_wire)

    pc(angles[0])
    for j in range(len(angles) - 1):
        if j % 2 == 0:
            apply_ua_fn()
        else:
            qml.adjoint(apply_ua_fn)()
        pc(angles[j + 1])


# Monotonic polynomial filter on [0, 1]: higher x => higher weight.
# We will sweep several even k values.
k_values = [8, 10, 12]
K_sweep_values = [0, 1, 2, 3, 4]
angles_monotonic = None
K = 0


def angles_for_k(k_power: int):
    p_monotonic = np.zeros(k_power + 1, dtype=float)
    p_monotonic[k_power] = 1.0
    return qml.poly_to_angles(p_monotonic, "QSVT")


def apply_A():
    init_pos_uniform()
    check_pos_compute()
    apply_qsvt_from_angles(angles_monotonic, apply_UA, work_wire=[sig_wire])


# -------------------------
# Success reflection + AA
# -------------------------


def S_success():
    # success condition uses sig=0; convert to control-on-1 with X.
    qml.PauliX(wires=sig_wire)
    qml.Toffoli(wires=[good_wire, sig_wire, flag_anc])
    qml.PauliZ(wires=flag_anc)
    qml.Toffoli(wires=[good_wire, sig_wire, flag_anc])
    qml.PauliX(wires=sig_wire)


def S0():
    # Memory-safe reflection about |0...0> on the A-subspace:
    # X^{\otimes} -> MCX to anc -> Z(anc) -> uncompute -> X^{\otimes}
    for w in a_wires:
        qml.PauliX(wires=w)

    qml.MultiControlledX(wires=a_wires + [diff_anc])
    qml.PauliZ(wires=diff_anc)
    qml.MultiControlledX(wires=a_wires + [diff_anc])

    for w in a_wires:
        qml.PauliX(wires=w)


def AA_step():
    S_success()
    qml.adjoint(apply_A)()
    S0()
    apply_A()


# -------------------------
# Choose K from p_success + sample after AA
# -------------------------


@qml.qnode(dev_ana)
def success_probs_after_A():
    apply_A()
    return qml.probs(wires=[good_wire, sig_wire])


@qml.qnode(dev_samp)
def sample_after_AA():
    apply_A()
    for _ in range(K):
        AA_step()
    return qml.sample(wires=time_wires)


def auto_K_from_success_prob(p_success: float):
    if p_success <= 0.0:
        return 0
    theta_success = math.asin(min(1.0, math.sqrt(p_success)))
    return int(math.floor((math.pi / (4.0 * theta_success)) - 0.5)) if theta_success > 0 else 0


def evaluate_config(k_power: int, K_iterations: int, p_success: float, K_auto: int, best_path_str: str):
    global K
    K = int(K_iterations)

    samples = sample_after_AA()

    rows = []
    best_measured = None
    for r in range(shots):
        path = decode_path_from_time_bits(samples[r])
        valid = int(is_valid_classical(path))

        if valid:
            c_norm = float(tour_cost_norm_from_indices(path))
            c_scaled = float((c_norm - cost_min) / cost_span)
            x = float(1.0 - c_scaled)
            if best_measured is None or c_norm < best_measured[0]:
                best_measured = (c_norm, path)
        else:
            c_norm = np.nan
            c_scaled = np.nan
            x = np.nan

        rows.append(
            {
                "path": path,
                "valid": valid,
                "cost_norm": c_norm,
                "cost_scaled_0_to_1": c_scaled,
                "quality_x_1_minus_scaled_cost": x,
            }
        )

    df = pd.DataFrame(rows)
    valid_rate = float(df["valid"].mean())

    valid_df = df[df["valid"] == 1].copy()
    if len(valid_df) > 0:
        valid_df["path_str"] = valid_df["path"].astype(str)
        ranking = (
            valid_df.groupby("path_str", as_index=False)
            .agg(
                count=("path_str", "size"),
                cost_norm=("cost_norm", "mean"),
                quality_x=("quality_x_1_minus_scaled_cost", "mean"),
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
        ranking = pd.DataFrame(
            columns=["path_str", "count", "cost_norm", "quality_x", "measured_rank", "classical_rank"]
        )

    best_row = ranking[ranking["path_str"] == best_path_str]
    if len(best_row) > 0:
        best_count = int(best_row.iloc[0]["count"])
        best_measured_rank = int(best_row.iloc[0]["measured_rank"])
    else:
        best_count = 0
        best_measured_rank = np.nan

    valid_count = int(df["valid"].sum())
    best_p_all = float(best_count / shots)
    best_p_valid = float(best_count / valid_count) if valid_count > 0 else np.nan
    mean_valid_cost = float(valid_df["cost_norm"].mean()) if len(valid_df) > 0 else np.nan

    summary_row = {
        "k_power": k_power,
        "K_iterations": int(K_iterations),
        "K_auto_from_p_success": int(K_auto),
        "p_success_before_AA": p_success,
        "valid_rate": valid_rate,
        "best_path": best_path_str,
        "best_path_count": best_count,
        "best_path_p_all_shots": best_p_all,
        "best_path_p_given_valid": best_p_valid,
        "best_path_measured_rank": best_measured_rank,
        "mean_valid_cost_norm": mean_valid_cost,
    }
    return summary_row, df, ranking, best_measured


def main():
    best_idx = int(np.argmax(x_vals))
    best_path_str = str(list(valid_tours[best_idx]))
    print("Best valid tour by classical cost:", list(valid_tours[best_idx]))
    print("Best-tour quality x:", float(x_vals[best_idx]))
    print("Running sweep for k values:", k_values)
    print("Running sweep for K values:", K_sweep_values)

    summary_rows = []
    best_payload = None  # (summary_row, df, ranking, best_measured)

    for k_power in k_values:
        global angles_monotonic
        angles_monotonic = angles_for_k(k_power)

        probs_gs = success_probs_after_A()
        p_success = float(probs_gs[2])
        K_auto = auto_K_from_success_prob(p_success)

        print("\n========================================")
        print(f"k = {k_power}")
        print("Estimated p_success after A (good=1 & sig=0):", p_success)
        print("Auto-chosen K from p_success =", K_auto)

        for K_candidate in K_sweep_values:
            row, df, ranking, best_measured = evaluate_config(
                k_power=k_power,
                K_iterations=K_candidate,
                p_success=p_success,
                K_auto=K_auto,
                best_path_str=best_path_str,
            )
            summary_rows.append(row)

            print(
                "k=", k_power,
                "K=", K_candidate,
                "valid_rate=", round(row["valid_rate"], 4),
                "best_p=", round(row["best_path_p_all_shots"], 4),
                "best_rank=", row["best_path_measured_rank"],
            )

            key = (row["best_path_p_all_shots"], row["valid_rate"])
            if best_payload is None:
                best_payload = (row, df, ranking, best_measured)
            else:
                best_key = (
                    best_payload[0]["best_path_p_all_shots"],
                    best_payload[0]["valid_rate"],
                )
                if key > best_key:
                    best_payload = (row, df, ranking, best_measured)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["best_path_p_all_shots", "valid_rate"], ascending=[False, False]
    ).reset_index(drop=True)
    kk_summary_out = "tsp_option1_kK_sweep_summary.csv"
    summary_df.to_csv(kk_summary_out, index=False)
    print("\nSaved:", kk_summary_out)

    # Keep the previous k-sweep summary name as "best K per each k" for compatibility.
    per_k_rows = []
    for k_power in k_values:
        per_k = summary_df[summary_df["k_power"] == k_power].copy()
        per_k = per_k.sort_values(
            ["best_path_p_all_shots", "valid_rate"], ascending=[False, False]
        ).reset_index(drop=True)
        per_k_rows.append(per_k.iloc[0])
    k_summary_df = pd.DataFrame(per_k_rows).sort_values(
        ["best_path_p_all_shots", "valid_rate"], ascending=[False, False]
    ).reset_index(drop=True)
    k_summary_out = "tsp_option1_k_sweep_summary.csv"
    k_summary_df.to_csv(k_summary_out, index=False)
    print("Saved:", k_summary_out)

    # Save the best configuration's detailed outputs.
    best_row, best_df, best_ranking, best_measured = best_payload
    best_k = int(best_row["k_power"])
    best_K = int(best_row["K_iterations"])
    best_samples_name = f"tsp_option1_best_k{best_k}_K{best_K}_samples.csv"
    best_ranking_name = f"tsp_option1_best_k{best_k}_K{best_K}_ranking.csv"
    best_df.to_csv(best_samples_name, index=False)
    best_ranking.to_csv(best_ranking_name, index=False)
    print("Saved:", best_samples_name)
    print("Saved:", best_ranking_name)

    # Backward-compatible aliases.
    best_df.to_csv("tsp_option1_monotonic_samples.csv", index=False)
    best_ranking.to_csv("tsp_option1_monotonic_ranking.csv", index=False)
    print("Saved:", "tsp_option1_monotonic_samples.csv")
    print("Saved:", "tsp_option1_monotonic_ranking.csv")

    print("\n========================================")
    print("Best configuration found:")
    print(
        "k=", best_k,
        "K=", best_K,
        "best_p=", round(float(best_row["best_path_p_all_shots"]), 6),
        "valid_rate=", round(float(best_row["valid_rate"]), 6),
        "best_rank=", best_row["best_path_measured_rank"],
    )
    if best_measured is not None:
        print("Best measured path in this config:", best_measured[1], "cost_norm:", best_measured[0])


if __name__ == "__main__":
    main()
# %%
