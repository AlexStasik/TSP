# %%
# ============================================================
# ARCHITEKTUR 1 (AA + "Hadamard-test cost to amplitude", NO QSVT)
#
# 1) Prepare uniform on time register
# 2) Grover AA to project onto valid tours (good)
# 3) Compute cost phase U_C: |p> -> e^{i C(p)} |p>  (controlled on good)
# 4) Hadamard test on ancilla 'succ' creates per-path success amplitude cos(C(p)/2)
# 5) Optional: Amplitude Amplification on success event succ=0
# 6) Sample and report top tours (conditioned on succ=0) + statevector amplitude ranking
# ============================================================

import math
import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf

# -------------------------
# Problem + cost matrix
# -------------------------
np.random.seed(42)

n = 5
start_node = n - 1

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)
C_max = float(np.max(all_costs_raw))
cost_matrix = cost_matrix_raw / C_max  # phases in ~[0,1]

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
state_wires = [f"s{i}" for i in range(n - 1)]   # parity flags
good_wire   = "good"
phase_anc   = "phase"
diff_anc    = "diff"
succ_wire   = "succ"

m = len(time_wires)
work_wires = [f"w{k}" for k in range(max(0, m - 2))]

wires = time_wires + state_wires + [good_wire, phase_anc, diff_anc, succ_wire] + work_wires

dev_sv = qml.device("lightning.qubit", wires=wires)  # analytic statevector
shots = 5000
dev_shots = qml.device("lightning.qubit", wires=wires, shots=shots)

# -------------------------
# Helpers: decode path, validity, cost (classical for reporting only)
# -------------------------
def decode_path_from_time_bits(bits):
    s = "".join(str(int(b)) for b in bits)
    path = []
    for t in range(T_steps):
        chunk = s[t * n_qubits_step : (t + 1) * n_qubits_step]
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
# Init: uniform on time register
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

def diffusion_time_register():
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

def grover_step_valid():
    phase_oracle_good()
    diffusion_time_register()

# Optimal K for validity AA
M = math.factorial(n - 1)
N = (n - 1) ** (n - 1)
p = M / N
theta = math.asin(math.sqrt(p))
K_valid = int(math.floor((math.pi / (4*theta)) - 0.5))
print("Validity AA: M =", M, "N =", N, "p =", p, "theta =", theta, "K_valid =", K_valid)

# -------------------------
# Cost phase oracle, controlled on good_wire
# This realizes U_C: valid path |p> -> e^{i C(p)} |p>, invalid unchanged.
# Uses phase kickback via toggling phase_anc.
# -------------------------
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

# -------------------------
# Hadamard-test mapping: per path success amplitude ~ cos(C/2) on succ=0
# Implements: H(succ) -> ctrl(U_C) -> H(succ)
# where ctrl is controlled by succ=1.
# -------------------------
def hadamard_test_cost_to_succ():
    qml.Hadamard(wires=succ_wire)
    qml.ctrl(cost_oracle_full_valid, control=succ_wire)()
    qml.Hadamard(wires=succ_wire)

# -------------------------
# A_total: prepare valid superposition + map cost to succ amplitude
# IMPORTANT: uncompute validity ancillas so AA reflections are valid.
# -------------------------
def apply_A_total():
    init_pos()
    for _ in range(K_valid):
        grover_step_valid()

    check_pos_compute()
    hadamard_test_cost_to_succ()
    qml.adjoint(check_pos_compute)()

# -------------------------
# AA on success event succ=0
# -------------------------
def S_success_succ0():
    # phase flip on |succ=0>
    qml.PauliX(wires=succ_wire)
    qml.PauliZ(wires=succ_wire)
    qml.PauliX(wires=succ_wire)

def S0_allzero():
    # reflection about |0...0> on full register
    qml.PCPhase(np.pi, dim=1, wires=wires)

def AA_step():
    S_success_succ0()
    qml.adjoint(apply_A_total)()
    S0_allzero()
    apply_A_total()

@qml.qnode(dev_sv)
def prob_succ_after_A():
    apply_A_total()
    return qml.probs(wires=[succ_wire])

p_succ = prob_succ_after_A()
p_success = float(p_succ[0])
phi = math.asin(min(1.0, math.sqrt(p_success)))
K_succ = int(math.floor((math.pi / (4*phi)) - 0.5)) if phi > 0 else 0

print("\nAfter A_total:")
print("p_success (succ=0) =", p_success)
print("AA iterations K_succ =", K_succ)

# -------------------------
# Statevector analysis: amplitudes conditioned on succ=0 (no sampling noise)
# -------------------------
@qml.qnode(dev_sv)
def state_after():
    apply_A_total()
    for _ in range(K_succ):
        AA_step()
    return qml.state()

state = state_after()

def bits_by_wire_from_index(idx):
    bitstr = format(idx, f"0{len(wires)}b")
    return {w: int(b) for w, b in zip(reversed(wires), reversed(bitstr))}

# Build table of amplitudes for succ=0 basis states
rows = []
amp_eps = 1e-12

for idx, amp in enumerate(state):
    a = complex(amp)
    prob = (a.real*a.real + a.imag*a.imag)
    if prob < amp_eps:
        continue

    bw = bits_by_wire_from_index(idx)
    succ = int(bw[succ_wire])
    if succ != 0:
        continue

    path = decode_path_from_time_bits([bw[w] for w in time_wires])
    valid = is_valid_classical(path)
    cost = classical_cost_norm(path) if valid else np.nan

    rows.append({
        "path": path,
        "valid": int(valid),
        "amp_abs": float(abs(a)),
        "prob": float(prob),
        "cost_norm": float(cost) if valid else np.nan,
    })

df_sv = pd.DataFrame(rows)
df_sv_sorted = df_sv.sort_values(["amp_abs"], ascending=False).reset_index(drop=True)

print("\nTop-20 states by |amplitude| conditioned on succ=0 (statevector):")
print(df_sv_sorted.head(20).to_string(index=False))

# -------------------------
# Sampling version (for realism): measure time_wires + succ
# -------------------------
@qml.qnode(dev_shots)
def sample_time_and_succ():
    apply_A_total()
    for _ in range(K_succ):
        AA_step()
    return qml.sample(wires=time_wires + [succ_wire])

samples = sample_time_and_succ()

rows = []
for r in range(shots):
    row = samples[r]
    time_bits = row[:-1]
    succ_bit = int(row[-1])
    path = decode_path_from_time_bits(time_bits)
    valid = is_valid_classical(path)
    cost = classical_cost_norm(path) if valid else np.nan
    rows.append({"path": path, "succ": succ_bit, "valid": int(valid), "cost_norm": cost})

df = pd.DataFrame(rows)
df0 = df[df["succ"] == 0].copy()

print("\nMeasured Pr(succ=0):", float((df["succ"] == 0).mean()))
print("Among succ=0, valid rate:", float(df0["valid"].mean()) if len(df0) else 0.0)

if len(df0):
    df0["path_str"] = df0["path"].astype(str)
    top = df0["path_str"].value_counts().head(15).reset_index()
    top.columns = ["path", "count"]
    top["cost_norm"] = top["path"].apply(lambda s: classical_cost_norm(eval(s)) if is_valid_classical(eval(s)) else np.nan)
    print("\nTop-15 paths among succ=0 (by count):")
    print(top.to_string(index=False))

out_path = "tsp_arch1_validAA_costAA.csv"
df.to_csv(out_path, index=False)
print("\nSaved:", out_path)
# %%