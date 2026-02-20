# %%
# ============================================================
# ARCHITEKTUR 2 (Valid-AA + QSVT on block-encoding, optional AA)
#
# 1) Validity Grover AA: project to valid
# 2) Build U_tilde:
#    valid -> e^{i C(p)}, invalid -> -1
# 3) Build UA block-encoding of A=(I+U_tilde)/2 using sig qubit:
#    In sig=0 block, singular value x(p)=|cos(theta(p)/2)|
#    invalid has theta=pi => x=0 exactly
# 4) Apply QSVT polynomial P(x)=x^k (kills x=0 exactly, boosts large x)
# 5) Optional AA on success event sig=0
# 6) Sample tours conditioned on sig=0 and compare costs
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
phase_anc   = "phase"
diff_anc    = "diff"
sig_wire    = "sig"
flag_anc    = "flag"

m = len(time_wires)
work_wires = [f"w{k}" for k in range(max(0, m - 2))]

wires = time_wires + state_wires + [good_wire, phase_anc, diff_anc, sig_wire, flag_anc] + work_wires

dev_sv = qml.device("lightning.qubit", wires=wires)
shots = 5000
dev_shots = qml.device("lightning.qubit", wires=wires, shots=shots)

# -------------------------
# Classical reporting helpers
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
# Init uniform
# -------------------------
def init_pos():
    for w in time_wires:
        qml.Hadamard(wires=w)

# -------------------------
# Validity oracle + Grover AA
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

M = math.factorial(n - 1)
N = (n - 1) ** (n - 1)
p = M / N
theta = math.asin(math.sqrt(p))
K_valid = int(math.floor((math.pi / (4*theta)) - 0.5))
print("Validity AA: M =", M, "N =", N, "p =", p, "theta =", theta, "K_valid =", K_valid)

# -------------------------
# Cost oracle controlled on good (same as Architektur 1)
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
# U_tilde: valid -> e^{iC}, invalid -> -1
# -------------------------
def invalid_phase_flip():
    # phase -1 on good=0
    qml.PauliX(wires=good_wire)
    qml.PauliZ(wires=good_wire)
    qml.PauliX(wires=good_wire)

def apply_U_tilde():
    check_pos_compute()
    invalid_phase_flip()
    cost_oracle_full_valid()
    qml.adjoint(check_pos_compute)()

# -------------------------
# UA: block-encoding of A=(I+U_tilde)/2 in sig=0 (Hadamard-control-Hadamard)
# -------------------------
def apply_UA():
    qml.Hadamard(wires=sig_wire)
    qml.ctrl(apply_U_tilde, control=sig_wire)()
    qml.Hadamard(wires=sig_wire)

# -------------------------
# QSVT core (P(x)=x^k)
# -------------------------
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

k_power = 12  # even; larger => stronger preference for large x
p_pow = np.zeros(k_power + 1, dtype=float)
p_pow[k_power] = 1.0
angles = qml.poly_to_angles(p_pow, "QSVT")

# -------------------------
# A_total: validity AA + QSVT(UA)
# IMPORTANT: uncompute validity ancillas? For QSVT we keep UA unitary; we run validity AA first,
# then QSVT. Validity AA leaves no ancillas dirty (it uses oracle that uncomputes).
# -------------------------
def apply_A_total():
    init_pos()
    for _ in range(K_valid):
        grover_step_valid()
    apply_qsvt_from_angles(angles, apply_UA, work_wire=[sig_wire])

# -------------------------
# Optional AA on success sig=0
# -------------------------
def S_success_sig0():
    qml.PauliX(wires=sig_wire)
    qml.PauliZ(wires=sig_wire)
    qml.PauliX(wires=sig_wire)

def S0_allzero():
    qml.PCPhase(np.pi, dim=1, wires=wires)

def AA_step():
    S_success_sig0()
    qml.adjoint(apply_A_total)()
    S0_allzero()
    apply_A_total()

@qml.qnode(dev_sv)
def prob_sig_after_A():
    apply_A_total()
    return qml.probs(wires=[sig_wire])

p_sig = prob_sig_after_A()
p_success = float(p_sig[0])
phi = math.asin(min(1.0, math.sqrt(p_success)))
K_sig = int(math.floor((math.pi / (4*phi)) - 0.5)) if phi > 0 else 0

print("\nAfter QSVT:")
print("p_success (sig=0) =", p_success)
print("AA iterations K_sig =", K_sig)

# -------------------------
# Sampling
# -------------------------
@qml.qnode(dev_shots)
def sample_time_and_sig():
    apply_A_total()
    for _ in range(K_sig):
        AA_step()
    return qml.sample(wires=time_wires + [sig_wire])

samples = sample_time_and_sig()

rows = []
for r in range(shots):
    row = samples[r]
    time_bits = row[:-1]
    sig_bit = int(row[-1])
    path = decode_path_from_time_bits(time_bits)
    valid = is_valid_classical(path)
    cost = classical_cost_norm(path) if valid else np.nan
    rows.append({"path": path, "sig": sig_bit, "valid": int(valid), "cost_norm": cost})

df = pd.DataFrame(rows)
df0 = df[df["sig"] == 0].copy()

print("\nMeasured Pr(sig=0):", float((df["sig"] == 0).mean()))
print("Among sig=0, valid rate:", float(df0["valid"].mean()) if len(df0) else 0.0)

if len(df0):
    df0["path_str"] = df0["path"].astype(str)
    top = df0["path_str"].value_counts().head(15).reset_index()
    top.columns = ["path", "count"]
    top["cost_norm"] = top["path"].apply(lambda s: classical_cost_norm(eval(s)) if is_valid_classical(eval(s)) else np.nan)
    print("\nTop-15 paths among sig=0 (by count):")
    print(top.to_string(index=False))

out_path = "tsp_arch2_validAA_qsvt_samples.csv"
df.to_csv(out_path, index=False)
print("\nSaved:", out_path)
# %%