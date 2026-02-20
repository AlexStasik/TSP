# %%
# ============================================================
# GROVER AA ONLY (VALIDITY) - STATEVECTOR VERSION
# - Prepare uniform over time strings
# - Grover AA on good states (valid tours)
# - Return full statevector
# - Analyze amplitudes/probabilities and decode top paths
# ============================================================

import math
import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf

# -------------------------
# Problem
# -------------------------
np.random.seed(42)

n = 5
start_node = n - 1

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)
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

time_wires  = [twire(t, q) for t in range(T_steps) for q in range(n_qubits_step)]
state_wires = [f"s{i}" for i in range(n - 1)]
good_wire   = "good"
diff_anc    = "diff"

m = len(time_wires)
work_wires = [f"w{k}" for k in range(max(0, m - 2))]

wires = time_wires + state_wires + [good_wire, diff_anc] + work_wires
dev = qml.device("lightning.qubit", wires=wires)

# -------------------------
# Counts and optimal K
# -------------------------
M = math.factorial(n - 1)
N = (n - 1) ** (n - 1)
p = M / N
theta = math.asin(math.sqrt(p))
K = int(math.floor((math.pi / (4 * theta)) - 0.5))

print("M =", M, "N =", N, "p =", p, "theta =", theta, "K =", K)

# -------------------------
# Init uniform on time register
# -------------------------
def init_pos():
    for w in time_wires:
        qml.Hadamard(wires=w)

# -------------------------
# Validity compute (your parity oracle)
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

def phase_oracle_good():
    check_pos_compute()
    qml.PauliZ(wires=good_wire)
    qml.adjoint(check_pos_compute)()

def diffusion_time():
    for w in time_wires:
        qml.Hadamard(wires=w)
    for w in time_wires:
        qml.PauliX(wires=w)

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
    diffusion_time()

# -------------------------
# Statevector circuit
# -------------------------
@qml.qnode(dev)
def state_circuit():
    init_pos()
    for _ in range(K):
        grover_step()
    # compute good for labeling/analysis (keine Uncompute nötig, wir analysieren state)
    check_pos_compute()
    return qml.state()

state = state_circuit()

# -------------------------
# Decode helpers
# -------------------------
def bits_by_wire_from_index(idx):
    bitstr = format(idx, f"0{len(wires)}b")
    # LSB corresponds to last wire; map robustly:
    return {w: int(b) for w, b in zip(reversed(wires), reversed(bitstr))}

def decode_path_from_bits(bits_by_wire):
    time_bits = "".join(str(bits_by_wire[w]) for w in time_wires)
    path = []
    for t in range(T_steps):
        chunk = time_bits[t * n_qubits_step : (t + 1) * n_qubits_step]
        path.append(int(chunk, 2))
    return path

def is_valid_classical(path):
    return sorted(path) == list(range(n - 1))

def classical_cost_norm(path):
    cost = float(cost_matrix[start_node, path[0]])
    for t in range(T_steps - 1):
        cost += float(cost_matrix[path[t], path[t+1]])
    cost += float(cost_matrix[path[-1], start_node])
    return float(cost)

# -------------------------
# Analyze full state: probabilities on good/valid, top amplitudes
# -------------------------
amp_eps = 1e-12
rows = []

p_total = 0.0
p_goodq = 0.0
p_validc = 0.0
p_good_and_valid = 0.0
mismatch_count = 0

for idx, amp in enumerate(state):
    a = complex(amp)
    prob = (a.real*a.real + a.imag*a.imag)
    if prob < amp_eps:
        continue

    bw = bits_by_wire_from_index(idx)
    path = decode_path_from_bits(bw)

    good_q = int(bw[good_wire])
    valid_c = int(is_valid_classical(path))

    p_total += prob
    if good_q == 1:
        p_goodq += prob
    if valid_c == 1:
        p_validc += prob
    if good_q == 1 and valid_c == 1:
        p_good_and_valid += prob
    if good_q != valid_c:
        mismatch_count += 1

    cost = classical_cost_norm(path) if valid_c else np.nan

    rows.append({
        "idx": idx,
        "path": path,
        "good_q": good_q,
        "valid_c": valid_c,
        "amp_abs": round(abs(a), 6),
        "prob": round(prob, 6),
        "amp_real": round(a.real, 6),
        "amp_imag": round(a.imag, 6),
        "cost_norm": round(float(cost), 6) if valid_c else np.nan,
    })

df = pd.DataFrame(rows)

print("\nStatevector prob mass (should be ~1):", p_total)
print("Prob mass good_q=1:", p_goodq)
print("Prob mass valid_c=1:", p_validc)
print("Prob mass (good_q=1 AND valid_c=1):", p_good_and_valid)
print("Mismatch basis states count (good_q != valid_c):", mismatch_count)

# Top amplitudes overall
df_top = df.sort_values("amp_abs", ascending=False).head(30).reset_index(drop=True)
print("\nTop-30 basis states by |amplitude|:")
print(df_top[["path","good_q","valid_c","amp_abs","prob","cost_norm"]].to_string(index=False))

# Top amplitudes restricted to valid (classical)
df_valid = df[df["valid_c"] == 1].sort_values("amp_abs", ascending=False).head(30).reset_index(drop=True)
print("\nTop-30 VALID basis states by |amplitude|:")
print(df_valid[["path","good_q","amp_abs","prob","cost_norm"]].to_string(index=False))

out_path = "tsp_grover_validity_statevector_table.csv"
df.sort_values(["valid_c","good_q","prob"], ascending=[False, False, False]).to_csv(out_path, index=False)
print("\nSaved:", out_path)
# %%