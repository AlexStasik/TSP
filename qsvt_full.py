# ============================================================
# qsvt_tsp_clean.py
# - Clean structure:
#   (1) Setup + classical helpers
#   (2) Quantum primitives (validity / cost / U_tilde / UA / QSVT)
#   (3) Benchmarks (encoding sanity)
#   (4) Experiment: Validity-AA -> QSVT -> statevector reports + rankings
# ============================================================

import math
import itertools
import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf

# -------------------------
# (1) Setup + cost scaling
# -------------------------
np.random.seed(42)

n = 5
start_node = n - 1

cost_matrix_raw = cf.generate_cost_matrix(n)

# Phase-range scaling: guarantee total tour phase theta(path) in [0, pi - margin]
margin = 1e-3
max_edge = float(np.max(cost_matrix_raw))
C_upper = float(n * max_edge)  # upper bound for full tour (n edges)
alpha = (np.pi - margin) / C_upper
cost_matrix = alpha * cost_matrix_raw  # entries are phases (radians)

# -------------------------
# (1b) Encoding (time-register)
# -------------------------
T_steps = n - 1
n_qubits_step = int(np.ceil(np.log2(n - 1)))

def twire(t, q):
    return f"t{t}_q{q}"

def pos_wires_at_t(t):
    return [twire(t, q) for q in range(n_qubits_step)]

time_wires  = [twire(t, q) for t in range(T_steps) for q in range(n_qubits_step)]
state_wires = [f"s{i}" for i in range(n - 1)]
good_wire   = "good"
phase_anc   = "phase"
sig_wire    = "sig"

m = len(time_wires)
work_wires = [f"w{k}" for k in range(max(0, m - 2))]

wires = time_wires + state_wires + [good_wire, phase_anc, sig_wire] + work_wires
dev = qml.device("lightning.qubit", wires=wires)  # analytic

# -------------------------
# (1c) Classical helpers
# -------------------------
def decode_path_from_time_bits(time_bits_as_ints):
    bits = "".join(str(int(b)) for b in time_bits_as_ints)
    path = []
    for t in range(T_steps):
        chunk = bits[t * n_qubits_step : (t + 1) * n_qubits_step]
        path.append(int(chunk, 2))
    return path

def is_valid_classical(path):
    return sorted(path) == list(range(n - 1))

def classical_theta(path):
    """Total tour phase theta(path) in radians, designed to lie in [0, pi-margin]."""
    theta = float(cost_matrix[start_node, path[0]])
    for t in range(T_steps - 1):
        theta += float(cost_matrix[path[t], path[t + 1]])
    theta += float(cost_matrix[path[-1], start_node])
    return float(theta)

def classical_cost_unit(path):
    """Unit cost in [0,1): theta/pi."""
    return float(classical_theta(path) / np.pi)

def classical_cost_raw(path):
    """Raw cost in original matrix."""
    cost = float(cost_matrix_raw[start_node, path[0]])
    for t in range(T_steps - 1):
        cost += float(cost_matrix_raw[path[t], path[t + 1]])
    cost += float(cost_matrix_raw[path[-1], start_node])
    return float(cost)

def performance_from_cost_unit(cost_u):
    """Simple linear performance: best=1, worst=0."""
    return float(1.0 - cost_u)

# ============================================================
# (2) Quantum primitives
# ============================================================

def init_pos():
    """Uniform superposition over ALL time-register bitstrings."""
    for w in time_wires:
        qml.Hadamard(wires=w)

def check_pos_compute():
    """
    Computes 'good' using your parity-style oracle:
    sets s_i flags, then MultiControlledX to good.
    """
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

def phase_oracle_good():
    """Marks good states by a phase flip (Grover oracle)."""
    check_pos_compute()
    qml.PauliZ(wires=good_wire)
    qml.adjoint(check_pos_compute)()

def diffusion_time_register():
    """Diffusion operator on the time register only."""
    for w in time_wires:
        qml.Hadamard(wires=w)
    for w in time_wires:
        qml.PauliX(wires=w)

    qml.MultiControlledX(
        wires=time_wires + [phase_anc],
        work_wires=work_wires,
        work_wire_type="zeroed",
    )
    qml.PauliZ(wires=phase_anc)
    qml.MultiControlledX(
        wires=time_wires + [phase_anc],
        work_wires=work_wires,
        work_wire_type="zeroed",
    )

    for w in time_wires:
        qml.PauliX(wires=w)
    for w in time_wires:
        qml.Hadamard(wires=w)

def grover_step_valid():
    phase_oracle_good()
    diffusion_time_register()

# ---- Cost oracle controlled on good (phase kickback on phase_anc) ----

def apply_transition_phase_valid(t):
    pos_t = pos_wires_at_t(t)
    pos_tp1 = pos_wires_at_t(t + 1)

    for i in range(n - 1):
        for j in range(n - 1):
            bits_i = format(i, f"0{n_qubits_step}b")
            bits_j = format(j, f"0{n_qubits_step}b")

            for w, b in zip(pos_t, bits_i):
                if b == "0":
                    qml.PauliX(wires=w)
            for w, b in zip(pos_tp1, bits_j):
                if b == "0":
                    qml.PauliX(wires=w)

            controls = [good_wire] + pos_t + pos_tp1
            qml.MultiControlledX(wires=controls + [phase_anc])
            qml.PhaseShift(float(cost_matrix[i, j]), wires=phase_anc)
            qml.MultiControlledX(wires=controls + [phase_anc])

            for w, b in zip(pos_tp1, bits_j):
                if b == "0":
                    qml.PauliX(wires=w)
            for w, b in zip(pos_t, bits_i):
                if b == "0":
                    qml.PauliX(wires=w)

def cost_oracle_full_valid():
    # Start -> first
    pos0 = pos_wires_at_t(0)
    for j in range(n - 1):
        bits = format(j, f"0{n_qubits_step}b")
        for w, b in zip(pos0, bits):
            if b == "0":
                qml.PauliX(wires=w)

        controls = [good_wire] + pos0
        qml.MultiControlledX(wires=controls + [phase_anc])
        qml.PhaseShift(float(cost_matrix[start_node, j]), wires=phase_anc)
        qml.MultiControlledX(wires=controls + [phase_anc])

        for w, b in zip(pos0, bits):
            if b == "0":
                qml.PauliX(wires=w)

    # Inner transitions
    for t in range(T_steps - 1):
        apply_transition_phase_valid(t)

    # Last -> start
    pos_last = pos_wires_at_t(T_steps - 1)
    for i in range(n - 1):
        bits = format(i, f"0{n_qubits_step}b")
        for w, b in zip(pos_last, bits):
            if b == "0":
                qml.PauliX(wires=w)

        controls = [good_wire] + pos_last
        qml.MultiControlledX(wires=controls + [phase_anc])
        qml.PhaseShift(float(cost_matrix[i, start_node]), wires=phase_anc)
        qml.MultiControlledX(wires=controls + [phase_anc])

        for w, b in zip(pos_last, bits):
            if b == "0":
                qml.PauliX(wires=w)

# ---- U_tilde & UA ----

def invalid_phase_flip():
    """Apply -1 phase on good=0."""
    qml.PauliX(wires=good_wire)
    qml.PauliZ(wires=good_wire)
    qml.PauliX(wires=good_wire)

def apply_U_tilde():
    """
    U_tilde:
      valid -> e^{i theta(path)}
      invalid -> -1
    plus uncompute validity ancillas.
    """
    check_pos_compute()
    invalid_phase_flip()
    cost_oracle_full_valid()
    qml.adjoint(check_pos_compute)()

def apply_UA():
    """
    Block-encoding UA of A=(I+U_tilde)/2 in sig=0:
    H(sig) - ctrl(U_tilde) - H(sig)
    """
    qml.Hadamard(wires=sig_wire)
    qml.ctrl(apply_U_tilde, control=sig_wire)()
    qml.Hadamard(wires=sig_wire)

# ---- QSVT: P(x)=x^k ----

def apply_qsvt_from_angles(angles, apply_UA_fn, work_wire):
    def PC(phi):
        qml.PCPhase(phi, dim=1, wires=work_wire)

    PC(angles[0])
    for j in range(len(angles) - 1):
        if j % 2 == 0:
            apply_UA_fn()
        else:
            qml.adjoint(apply_UA_fn)()
        PC(angles[j + 1])

k_power = 12
p_pow = np.zeros(k_power + 1, dtype=float)
p_pow[k_power] = 1.0
angles = qml.poly_to_angles(p_pow, "QSVT")

# ============================================================
# (3) Benchmarks / tests
# ============================================================

def wrap_to_pi(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)

def bits_by_wire_from_index(idx: int):
    bitstr = format(idx, f"0{len(wires)}b")
    # LSB = last wire in 'wires'
    return {w: int(b) for w, b in zip(reversed(wires), reversed(bitstr))}

def decode_path_from_bw(bw):
    return decode_path_from_time_bits([bw[w] for w in time_wires])

def ancillas_all_zero_bw(bw):
    """All non-time wires are 0 (state_wires + good + phase + sig + work)."""
    for w in wires:
        if w in time_wires:
            continue
        if bw[w] != 0:
            return False
    return True

@qml.qnode(dev)
def state_validity_only():
    init_pos()
    check_pos_compute()
    return qml.state()

@qml.qnode(dev)
def state_cost_only():
    init_pos()
    check_pos_compute()
    cost_oracle_full_valid()
    return qml.state()

@qml.qnode(dev)
def state_utilde_only():
    init_pos()
    apply_U_tilde()
    return qml.state()

def run_benchmarks(eps_amp: float = 1e-12):
    print("\n==============================")
    print("RUNNING BENCHMARKS")
    print("==============================")

    # [B0] Phase-range bound check (benchmark enumeration of valid tours)
    valid_paths = list(itertools.permutations(range(n - 1), r=(n - 1)))
    thetas = [classical_theta(list(p)) for p in valid_paths]
    th_min = float(np.min(thetas))
    th_max = float(np.max(thetas))
    violations = sum(1 for th in thetas if not (0.0 <= th <= (np.pi - margin + 1e-12)))

    print("\n[Benchmark 0] Phase-range bound check (valid paths enumerated)")
    print("min theta:", th_min)
    print("max theta:", th_max)
    print("violations:", violations)

    # [B1] Validity oracle vs classical validity
    st = state_validity_only()
    mismatches = 0
    count_considered = 0

    for idx, amp in enumerate(st):
        if abs(amp) < eps_amp:
            continue
        bw = bits_by_wire_from_index(idx)
        path = decode_path_from_bw(bw)
        good_q = int(bw[good_wire])
        valid_c = int(is_valid_classical(path))
        count_considered += 1
        mismatches += int(good_q != valid_c)

    print("\n[Benchmark 1] Validity oracle vs classical validity")
    print("basis states with |amp|>eps:", count_considered)
    print("mismatches good != valid:", mismatches)

    # [B2] Cost-phase oracle consistency (valid, good=1, phase_anc=0)
    st2 = state_cost_only()
    ref_idx = None
    ref_phase = None
    ref_theta = None

    for idx, amp in enumerate(st2):
        if abs(amp) < eps_amp:
            continue
        bw = bits_by_wire_from_index(idx)
        if int(bw[good_wire]) != 1 or int(bw[phase_anc]) != 0:
            continue
        path = decode_path_from_bw(bw)
        if not is_valid_classical(path):
            continue
        ref_idx = idx
        ref_phase = float(np.angle(amp))
        ref_theta = float(classical_theta(path))
        break

    print("\n[Benchmark 2] Cost-phase oracle consistency (cost-only circuit)")
    if ref_idx is None:
        print("No valid reference found (try lowering eps_amp).")
    else:
        worst_err = 0.0
        checked = 0
        for idx, amp in enumerate(st2):
            if abs(amp) < eps_amp:
                continue
            bw = bits_by_wire_from_index(idx)
            if int(bw[good_wire]) != 1 or int(bw[phase_anc]) != 0:
                continue
            path = decode_path_from_bw(bw)
            if not is_valid_classical(path):
                continue

            phase = float(np.angle(amp))
            dphi = wrap_to_pi(phase - ref_phase)

            th = float(classical_theta(path))
            dth = wrap_to_pi(th - ref_theta)

            err = abs(wrap_to_pi(dphi - dth))
            worst_err = max(worst_err, err)
            checked += 1

        print("reference idx:", ref_idx)
        print("checked valid states:", checked)
        print("max |wrap(dphi - dtheta)|:", worst_err)

    # [B3] U_tilde ancilla reset and invalid=-1 consistency (ancillas=0 subspace)
    st3 = state_utilde_only()

    total_prob = 0.0
    prob_anc0 = 0.0
    for idx, amp in enumerate(st3):
        p = float(abs(amp) ** 2)
        total_prob += p
        bw = bits_by_wire_from_index(idx)
        if ancillas_all_zero_bw(bw):
            prob_anc0 += p

    print("\n[Benchmark 3] U_tilde ancilla reset / leakage check")
    print("Total prob mass:", total_prob)
    print("Prob mass with all ancillas=0:", prob_anc0)
    print("Leakage mass:", total_prob - prob_anc0)

    ref_phase = None
    ref_theta = None
    ref_path = None
    for idx, amp in enumerate(st3):
        if abs(amp) < eps_amp:
            continue
        bw = bits_by_wire_from_index(idx)
        if not ancillas_all_zero_bw(bw):
            continue
        path = decode_path_from_bw(bw)
        if is_valid_classical(path):
            ref_phase = float(np.angle(amp))
            ref_theta = float(classical_theta(path))
            ref_path = path
            break

    print("\n[Benchmark 3] U_tilde invalid=-1 consistency (ancillas=0 subspace)")
    if ref_phase is None:
        print("No valid reference found in ancillas=0 subspace (unexpected).")
    else:
        worst_err = 0.0
        worst_path = None
        checked = 0
        invalid_printed = 0

        for idx, amp in enumerate(st3):
            if abs(amp) < eps_amp:
                continue
            bw = bits_by_wire_from_index(idx)
            if not ancillas_all_zero_bw(bw):
                continue

            path = decode_path_from_bw(bw)
            valid = is_valid_classical(path)

            phase = float(np.angle(amp))
            dphi = wrap_to_pi(phase - ref_phase)

            if valid:
                expected = wrap_to_pi(float(classical_theta(path) - ref_theta))
            else:
                expected = wrap_to_pi(np.pi - ref_theta)

            err = abs(wrap_to_pi(dphi - expected))
            checked += 1
            if err > worst_err:
                worst_err = err
                worst_path = path

            if (not valid) and invalid_printed < 5:
                print("\ninvalid example:", path)
                print("  dphi:", dphi, " expected:", expected, " err:", err)
                invalid_printed += 1

        print("ref_path:", ref_path, "ref_theta:", ref_theta)
        print("checked time-basis states (ancillas=0):", checked)
        print("worst err:", worst_err, "worst_path:", worst_path)

    print("\n==============================")
    print("BENCHMARKS DONE")
    print("==============================\n")

# ============================================================
# (4) Experiment: Validity-AA -> QSVT -> statevector reports
# ============================================================

# Validity AA iteration count (you know M,N exactly here)
M = math.factorial(n - 1)
N = (n - 1) ** (n - 1)
p_good = M / N
theta_g = math.asin(math.sqrt(p_good))
K_valid = int(math.floor((math.pi / (4 * theta_g)) - 0.5))

print(f"Validity AA: M={M} N={N} p={p_good} theta={theta_g} K_valid={K_valid}")

def apply_A_total():
    init_pos()
    for _ in range(K_valid):
        grover_step_valid()
    apply_qsvt_from_angles(angles, apply_UA, work_wire=[sig_wire])

@qml.qnode(dev)
def probs_sig():
    apply_A_total()
    return qml.probs(wires=[sig_wire])

@qml.qnode(dev)
def state_after():
    apply_A_total()
    return qml.state()

def report_for_sig(state, sig_value: int, topk: int = 30, amp_eps: float = 1e-15):
    rows = []
    p_sig_total = 0.0

    for idx, amp in enumerate(state):
        if abs(amp) < amp_eps:
            continue

        bw = bits_by_wire_from_index(idx)
        if int(bw[sig_wire]) != int(sig_value):
            continue

        path = decode_path_from_bw(bw)
        valid = is_valid_classical(path)

        prob = float(abs(amp) ** 2)
        p_sig_total += prob

        if valid:
            th = float(classical_theta(path))
            cu = float(classical_cost_unit(path))
            perf = float(performance_from_cost_unit(cu))
            x = float(abs(np.cos(th / 2.0)))
            pred_weight = float(x ** (2 * k_power))
        else:
            th = np.nan
            cu = np.nan
            perf = np.nan
            x = np.nan
            pred_weight = np.nan

        rows.append({
            "path": path,
            "valid": int(valid),
            "amp_abs": float(abs(amp)),
            "prob_joint": prob,
            "theta(rad)": th,
            "cost_unit": cu,
            "performance": perf,
            "x=|cos(theta/2)|": x,
            "pred_weight~x^(2k)": pred_weight,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No amplitude mass found in sig={sig_value}.")

    # Conditional prob within sig-subspace
    df[f"prob_cond(path | sig={sig_value})"] = df["prob_joint"] / p_sig_total

    # quantum rank
    df["quantum_rank"] = df[f"prob_cond(path | sig={sig_value})"].rank(
        ascending=False, method="dense"
    ).astype(int)

    # true rank among valid only (ascending cost_unit)
    df["true_rank"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    mv = df["valid"] == 1
    df.loc[mv, "true_rank"] = df.loc[mv, "cost_unit"].rank(
        ascending=True, method="dense"
    ).astype("Int64")

    # predicted rank among valid only (descending pred_weight)
    df["pred_rank"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df.loc[mv, "pred_rank"] = df.loc[mv, "pred_weight~x^(2k)"].rank(
        ascending=False, method="dense"
    ).astype("Int64")

    # sort
    df_sorted = df.sort_values(["quantum_rank", "true_rank"], ascending=[True, True]).reset_index(drop=True)

    # exact valid rate inside sig-subspace
    valid_rate = float(df.loc[df["valid"] == 1, "prob_joint"].sum() / p_sig_total)

    # rounding for readability
    round_cols = [
        "amp_abs", "prob_joint", f"prob_cond(path | sig={sig_value})",
        "theta(rad)", "cost_unit", "performance",
        "x=|cos(theta/2)|", "pred_weight~x^(2k)"
    ]
    for c in round_cols:
        df_sorted[c] = df_sorted[c].astype(float).round(12)

    print(f"\n=== Statevector report conditioned on sig={sig_value} ===")
    print("Total prob mass in this sig subspace:", p_sig_total)
    print("Valid rate in this sig subspace (exact):", valid_rate)
    print(f"\nTop-{topk} by quantum_rank:")
    print(
        df_sorted[
            ["quantum_rank", "true_rank", "pred_rank",
             "path", "valid",
             f"prob_cond(path | sig={sig_value})",
             "theta(rad)", "cost_unit", "performance",
             "x=|cos(theta/2)|", "pred_weight~x^(2k)"]
        ].head(topk).to_string(index=False)
    )

    out = f"tsp_qsvt_statevector_cond_sig{sig_value}.csv"
    df_sorted.to_csv(out, index=False)
    print("Saved:", out)

    return df_sorted, p_sig_total, valid_rate

def main():
    # 1) Benchmarks first (sanity)
    run_benchmarks(eps_amp=1e-12)

    # 2) Run experiment
    p = probs_sig()
    print("\nAfter QSVT:")
    print("p_success (sig=0) =", float(p[0]))

    st = state_after()
    report_for_sig(st, 0, topk=30)
    report_for_sig(st, 1, topk=30)

if __name__ == "__main__":
    main()