# %%
# ============================================================
# GROVER FILTER (VALID PATHS) + COST PHASE CHECK (AFTER ONLY)
# - Grover AA amplifies valid paths
# - then applies full cost phase oracle
# - compares measured relative phases vs classical normalized cost
# ============================================================

import math
import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf

# -------------------------
# Problem + cost matrix
# -------------------------
n = 5
start_node = n-1

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)

C_max = np.max(all_costs_raw)
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

time_wires  = [twire(t, q) for t in range(T_steps) for q in range(n_qubits_step)]
state_wires = [f"s{i}" for i in range(n - 1)]

good_wire = "good"
diff_anc  = "diff"
phase_anc = "phase"

m = len(time_wires)
N = 2**m
M = math.factorial(n - 1)        # number of valid permutations of length (n-1)
p = M / N
theta = math.asin(math.sqrt(p))
K = int(math.floor((math.pi / (4*theta)) - 0.5))

print("m =", m, "N =", N, "M =", M, "p =", p, "theta =", theta, "K =", K)

# work wires for diffusion MCX (m controls)
work_wires = [f"w{k}" for k in range(max(0, m - 2))]

wires = time_wires + state_wires + [good_wire, diff_anc, phase_anc] + work_wires
dev = qml.device("lightning.qubit", wires=wires)

# -------------------------
# Init
# -------------------------
def init_pos():
    for w in time_wires:
        qml.Hadamard(wires=w)

# -------------------------
# Validity compute (writes parity flags into state_wires, then ANDs into good_wire)
# NOTE: This is the same compute used inside the Grover phase oracle.
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
# Grover phase oracle for "good" subspace: |x>|0...0> -> (-1) if good(x)
# We compute good, Z, then uncompute so ancillas return to |0>.
# -------------------------
def phase_oracle_good():
    check_pos_compute()
    qml.PauliZ(wires=good_wire)
    qml.adjoint(check_pos_compute)()

# -------------------------
# Diffusion on time register (m qubits) using diff_anc + work_wires
# -------------------------
def diffusion_time_register():
    for w in time_wires:
        qml.Hadamard(wires=w)
    for w in time_wires:
        qml.PauliX(wires=w)

    # implement Z on |11..1> of time_wires via compute-to-anc, Z(anc), uncompute
    qml.MultiControlledX(
        wires=time_wires + [diff_anc],
        work_wires=work_wires,
        work_wire_type="zeroed",
    )
    qml.PauliZ(wires=diff_anc)
    qml.MultiControlledX(
        wires=time_wires + [diff_anc],
        work_wires=work_wires,
        work_wire_type="zeroed",
    )

    for w in time_wires:
        qml.PauliX(wires=w)
    for w in time_wires:
        qml.Hadamard(wires=w)

def grover_step():
    phase_oracle_good()
    diffusion_time_register()

# -------------------------
# FULL cost oracle (Start + transitions + return)
# PhaseShift angles are in [0,1] radians by your normalization.
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
            qml.PhaseShift(cost_matrix[i, j], wires=phase_anc)
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
        qml.PhaseShift(cost_matrix[start_node, j], wires=phase_anc)
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
        qml.PhaseShift(cost_matrix[i, start_node], wires=phase_anc)
        qml.MultiControlledX(wires=pos_last + [phase_anc])

        for w, b in zip(pos_last, bits):
            if b == "0":
                qml.PauliX(wires=w)

# -------------------------
# QNode (AFTER ONLY)
# -------------------------
@qml.qnode(dev)
def circuit_after():
    init_pos()
    for _ in range(K):
        grover_step()

    # optional: compute good again for logging (does NOT affect probs; only flips ancillas)
    check_pos_compute()

    # encode cost phase AFTER AA
    cost_oracle_full()
    return qml.state()

state = circuit_after()

# -------------------------
# Decode helpers (robust to wire order)
# -------------------------
def bits_by_wire_from_index(idx):
    bitstr = format(idx, f"0{len(wires)}b")
    return bitstr, {w: int(b) for w, b in zip(reversed(wires), reversed(bitstr))}

def decode_path_from_bits(bits_by_wire):
    time_bits = "".join(str(bits_by_wire[w]) for w in time_wires)
    path = []
    for t in range(T_steps):
        chunk = time_bits[t*n_qubits_step:(t+1)*n_qubits_step]
        path.append(int(chunk, 2))
    return path

def is_valid_classical(path):
    return sorted(path) == list(range(n - 1))

def classical_cost_norm(path):
    cost = cost_matrix[start_node, path[0]]
    for t in range(T_steps - 1):
        cost += cost_matrix[path[t], path[t+1]]
    cost += cost_matrix[path[-1], start_node]
    return cost

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

# -------------------------
# Build table + phase check
# -------------------------
def build_table_with_phase_check(state):
    # reference among valid+good states
    ref_phase = None
    ref_cost = None

    for idx, amp in enumerate(state):
        if abs(amp) < 1e-12:
            continue
        _, bw = bits_by_wire_from_index(idx)
        path = decode_path_from_bits(bw)
        if int(bw[good_wire]) == 1 and is_valid_classical(path):
            ref_phase = np.angle(amp)
            ref_cost = classical_cost_norm(path)
            break

    if ref_phase is None:
        raise RuntimeError("No valid+good reference state found.")

    rows = []
    for idx, amp in enumerate(state):
        if abs(amp) < 1e-12:
            continue

        bitstr, bw = bits_by_wire_from_index(idx)
        path = decode_path_from_bits(bw)

        good_q = int(bw[good_wire])
        valid_c = is_valid_classical(path)

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
            "amp_abs": float(abs(amp)),
            "prob": float(abs(amp)**2),
            "phase": phase,
            "dphi": dphi,
            "cost_norm": cost,
            "dcost": dcost,
            "diff_wrapped": diff_wrapped,
        })

    return pd.DataFrame(rows)

dfa = build_table_with_phase_check(state)

# -------------------------
# Report + save
# -------------------------
p_good = dfa.loc[dfa["is_valid_classical"]==1, "prob"].sum()
p_bad  = dfa.loc[dfa["is_valid_classical"]==0, "prob"].sum()

print("Success prob (valid/classical):", p_good)
print("Bad prob mass:", p_bad)
print("Total prob mass (rows summed):", p_good + p_bad)

mismatch = dfa[dfa["good_quantum"] != dfa["is_valid_classical"]]
print("mismatches:", len(mismatch))

max_diff = np.nanmax(dfa.loc[dfa["is_valid_classical"]==1, "diff_wrapped"].values)
print("max diff_wrapped (valid only):", max_diff)

out_path = "tsp_grover_filter_after_only.csv"
dfa.to_csv(out_path, index=False)
print("Saved:", out_path)

# display(dfa.sort_values(["good_quantum","prob"], ascending=[False, False]).head(30))
# %%