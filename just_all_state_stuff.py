# %%
# ============================================================
# BASELINE: ALL WALKS (uniform superposition) +
#           VALIDITY ORACLE (parity-based) +
#           FULL COST PHASE ORACLE
# Then: statevector analysis + classical comparison table
#
# No amplitude amplification, no QSVT.
# ============================================================

import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf

# -------------------------
# Problem + cost matrix
# -------------------------
n = 5
start_node = n - 1

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)

C_max = float(np.max(all_costs_raw))
cost_matrix = cost_matrix_raw / C_max  # entries are phases in radians in [0, ~1]

# -------------------------
# Encoding (time-register)
# -------------------------
T_steps = n - 1
n_qubits_step = int(np.ceil(np.log2(n - 1)))

def twire(t, q):
    return f"t{t}_q{q}"

def pos_wires_at_t(t):
    return [twire(t, q) for q in range(n_qubits_step)]

time_wires  = [twire(t, q) for t in range(T_steps) for q in range(n_qubits_step)]
state_wires = [f"s{i}" for i in range(n - 1)]  # parity flags
good_wire   = "good"
phase_anc   = "phase"

# Optional work wires for big MCX; not required here (controls are small), but kept consistent
m = len(time_wires)
work_wires = [f"w{k}" for k in range(max(0, m - 2))]

wires = time_wires + state_wires + [good_wire, phase_anc] + work_wires
dev = qml.device("lightning.qubit", wires=wires)

# -------------------------
# Init: uniform over ALL time-register bitstrings
# -------------------------
def init_pos():
    for w in time_wires:
        qml.Hadamard(wires=w)

# -------------------------
# Validity compute (your parity-based oracle)
# Writes parity flags into state_wires and ANDs into good_wire
# -------------------------
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
# FULL cost oracle (Start + transitions + return)
# Uses phase kickback via toggling phase_anc
# -------------------------
def apply_transition_phase(t):
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

            controls = pos_t + pos_tp1
            qml.MultiControlledX(wires=controls + [phase_anc])
            qml.PhaseShift(float(cost_matrix[i, j]), wires=phase_anc)
            qml.MultiControlledX(wires=controls + [phase_anc])

            for w, b in zip(pos_tp1, bits_j):
                if b == "0":
                    qml.PauliX(wires=w)
            for w, b in zip(pos_t, bits_i):
                if b == "0":
                    qml.PauliX(wires=w)

def cost_oracle_full():
    # Start -> first
    pos0 = pos_wires_at_t(0)
    for j in range(n - 1):
        bits = format(j, f"0{n_qubits_step}b")

        for w, b in zip(pos0, bits):
            if b == "0":
                qml.PauliX(wires=w)

        qml.MultiControlledX(wires=pos0 + [phase_anc])
        qml.PhaseShift(float(cost_matrix[start_node, j]), wires=phase_anc)
        qml.MultiControlledX(wires=pos0 + [phase_anc])

        for w, b in zip(pos0, bits):
            if b == "0":
                qml.PauliX(wires=w)

    # Inner transitions
    for t in range(T_steps - 1):
        apply_transition_phase(t)

    # Last -> start
    pos_last = pos_wires_at_t(T_steps - 1)
    for i in range(n - 1):
        bits = format(i, f"0{n_qubits_step}b")

        for w, b in zip(pos_last, bits):
            if b == "0":
                qml.PauliX(wires=w)

        qml.MultiControlledX(wires=pos_last + [phase_anc])
        qml.PhaseShift(float(cost_matrix[i, start_node]), wires=phase_anc)
        qml.MultiControlledX(wires=pos_last + [phase_anc])

        for w, b in zip(pos_last, bits):
            if b == "0":
                qml.PauliX(wires=w)

# -------------------------
# QNode: Uniform -> validity compute -> cost phase
# (No Grover, no QSVT)
# -------------------------
@qml.qnode(dev)
def circuit_baseline():
    init_pos()
    check_pos_compute()   # marks allowed walks into good_wire (and sets state_wires)
    cost_oracle_full()    # encodes loss as phase
    return qml.state()

state = circuit_baseline()

# -------------------------
# Decode helpers (robust to wire order)
# -------------------------
def bits_by_wire_from_index(idx):
    bitstr = format(idx, f"0{len(wires)}b")
    # PennyLane uses wires list ordering; map bitstring to wires robustly:
    # The least-significant bit corresponds to the last wire in 'wires'.
    return bitstr, {w: int(b) for w, b in zip(reversed(wires), reversed(bitstr))}

def decode_path_from_bits(bits_by_wire):
    time_bits = "".join(str(bits_by_wire[w]) for w in time_wires)
    path = []
    for t in range(T_steps):
        chunk = time_bits[t * n_qubits_step : (t + 1) * n_qubits_step]
        path.append(int(chunk, 2))
    return path

def is_valid_classical(path):
    # valid iff it's a permutation of 0..n-2
    return sorted(path) == list(range(n - 1))

def classical_cost_norm(path):
    # cost in normalized matrix
    cost = float(cost_matrix[start_node, path[0]])
    for t in range(T_steps - 1):
        cost += float(cost_matrix[path[t], path[t+1]])
    cost += float(cost_matrix[path[-1], start_node])
    return float(cost)

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

# -------------------------
# Build table + phase check
# We use one valid state as reference to compare relative phases:
# dphi(pi) should match dcost(pi) (mod 2pi) for valid paths.
# -------------------------
def build_table_with_phase_check(statevec, amp_eps=1e-12):
    ref_phase = None
    ref_cost = None

    # find a reference VALID state with good=1 (and phase_anc=0 ideally)
    for idx, amp in enumerate(statevec):
        if abs(amp) < amp_eps:
            continue
        _, bw = bits_by_wire_from_index(idx)
        path = decode_path_from_bits(bw)

        good_q = int(bw[good_wire])
        valid_c = is_valid_classical(path)
        phase_bit = int(bw[phase_anc])

        if good_q == 1 and valid_c and phase_bit == 0:
            ref_phase = float(np.angle(amp))
            ref_cost = float(classical_cost_norm(path))
            break

    if ref_phase is None:
        raise RuntimeError("No valid+good reference state found (try lowering amp_eps).")

    rows = []
    for idx, amp in enumerate(statevec):
        if abs(amp) < amp_eps:
            continue

        bitstr, bw = bits_by_wire_from_index(idx)
        path = decode_path_from_bits(bw)

        good_q = int(bw[good_wire])
        valid_c = is_valid_classical(path)
        phase_bit = int(bw[phase_anc])

        phase = float(np.angle(amp))
        dphi = float(wrap_to_pi(phase - ref_phase))

        if valid_c:
            cost = float(classical_cost_norm(path))
            dcost = float(cost - ref_cost)
            diff_wrapped = float(abs(wrap_to_pi(dphi - dcost)))
        else:
            cost = np.nan
            dcost = np.nan
            diff_wrapped = np.nan

        rows.append({
            "idx": idx,
            "bitstr": bitstr,
            "path": path,
            "good_quantum": good_q,
            "is_valid_classical": int(valid_c),
            "phase_anc_bit": phase_bit,
            "amp_abs": float(abs(amp)),
            "prob": float(abs(amp)**2),
            "phase": phase,
            "dphi": dphi,
            "cost_norm": cost,
            "dcost": dcost,
            "diff_wrapped": diff_wrapped,
        })

    return pd.DataFrame(rows)

df = build_table_with_phase_check(state)

# -------------------------
# Report + save
# -------------------------
# Compare quantum good flag vs classical validity
mismatch = df[df["good_quantum"] != df["is_valid_classical"]]
print("mismatches (good_quantum != is_valid_classical):", len(mismatch))

# Probability mass by classical validity (should be near 1 on all states because uniform;
# but note: we're summing only populated basis states; should sum to 1)
p_valid = float(df.loc[df["is_valid_classical"] == 1, "prob"].sum())
p_invalid = float(df.loc[df["is_valid_classical"] == 0, "prob"].sum())
print("Prob mass valid:", p_valid)
print("Prob mass invalid:", p_invalid)
print("Total prob mass (rows summed):", p_valid + p_invalid)

# Phase consistency on valid states
max_diff = float(np.nanmax(df.loc[df["is_valid_classical"] == 1, "diff_wrapped"].values))
print("max |wrap(dphi - dcost)| over valid states:", max_diff)

# Sort for convenience: valid & good on top, then by cost
df_sorted = df.sort_values(
    ["is_valid_classical", "good_quantum", "cost_norm"],
    ascending=[False, False, True],
).reset_index(drop=True)

out_path = "tsp_baseline_validity_cost_phase.csv"
df_sorted.to_csv(out_path, index=False)
print("Saved:", out_path)

# optional preview
print("\nTop-20 valid states (by increasing cost_norm):")
print(df_sorted[df_sorted["is_valid_classical"] == 1][["path","good_quantum","cost_norm","phase","diff_wrapped"]].head(20).to_string(index=False))
# %%