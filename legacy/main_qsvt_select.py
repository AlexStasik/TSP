# %%
# ============================================================
# VALIDITY (good) + QSVT COST FILTER (sig) + ONE AA on (good=1 AND sig=0)
#
# - Uses your parity-based validity compute (check_pos_compute)
# - Builds a diagonal block-encoding UA by lookup over VALID tours:
#     for each valid tour bitstring s: Ry(2*theta_s) on sig, where cos(theta_s)=x(s) in [-1,1]
# - Applies QSVT polynomial P(x) (even/cosh filter) on the block-encoding
# - Defines Success := (good=1) AND (sig=0)
# - Runs one amplitude amplification (Grover-style) for Success
# - Samples time register; valid tours dominate; shorter tours have higher weight
#
# NOTES:
# - Simulator-friendly, not scalable: the block-encoding UA loops over all (n-1)! valid tours.
# - "Safety / range-check for invalid values" is intentionally NOT implemented.
# ============================================================

import math
import itertools
import numpy as np
import pandas as pd
import pennylane as qml
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P
import src.classical_funcs as cf

# -------------------------
# Problem + normalized cost matrix
# -------------------------
n = 5
start_node = n - 1

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)
C_max = float(np.max(all_costs_raw))
cost_matrix = cost_matrix_raw / C_max

# -------------------------
# Encoding (time registers)
# -------------------------
T_steps = n - 1
n_qubits_step = int(np.ceil(np.log2(n - 1)))

def twire(t, q):
    return f"t{t}_q{q}"

def pos_wires_at_t(t):
    return [twire(t, q) for q in range(n_qubits_step)]

time_wires = [twire(t, q) for t in range(T_steps) for q in range(n_qubits_step)]
state_wires = [f"s{i}" for i in range(n - 1)]  # parity flags per node

good_wire = "good"
sig_wire  = "sig"
diff_anc  = "diff"
flag_anc  = "flag"  # used only to implement the Success phase flip cleanly

m = len(time_wires)
work_wires = [f"w{k}" for k in range(max(0, m - 2))]

wires = time_wires + state_wires + [good_wire, sig_wire, diff_anc, flag_anc] + work_wires

# We'll use two devices:
# - an analytic device to estimate p_success and choose K
# - a shots device to sample tours after AA
dev_ana = qml.device("lightning.qubit", wires=wires)
shots = 5000
dev_samp = qml.device("lightning.qubit", wires=wires, shots=shots)

# -------------------------
# Helpers: bits and indexing (MSB->LSB consistent with time_wires order)
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

# -------------------------
# Init: uniform over time register
# -------------------------
def init_pos_uniform():
    for w in time_wires:
        qml.Hadamard(wires=w)

# -------------------------
# Your validity compute: parity flags into state_wires, then AND into good_wire
# (Assumed correct in your setting, as discussed.)
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
# Classical enumeration of VALID tours for the block-encoding lookup (small-n)
# We encode non-start nodes as values 0..n-2 directly in the time registers.
# -------------------------
def all_valid_tours_indices(n_):
    for perm in itertools.permutations(range(n_ - 1)):
        yield perm  # tuple length n-1 over {0..n-2}

def tour_cost_norm_from_indices(path_idx):
    # path_idx are indices 0..n-2 which correspond to actual nodes (same indices),
    # start_node is n-1.
    c = float(cost_matrix[start_node, path_idx[0]])
    for t in range(T_steps - 1):
        c += float(cost_matrix[path_idx[t], path_idx[t + 1]])
    c += float(cost_matrix[path_idx[-1], start_node])
    return float(c)

valid_tours = list(all_valid_tours_indices(n))
valid_costs = np.array([tour_cost_norm_from_indices(p) for p in valid_tours], dtype=float)

# map cost_norm in [0, ?] -> compress to [0,1] by dividing by max over valid tours (safe)
Cmax_valid = float(np.max(valid_costs))
costs_norm = valid_costs / (Cmax_valid if Cmax_valid > 0 else 1.0)

# quality signal x in [-1,1], x=1 is best
x_vals = 1.0 - 2.0 * costs_norm
thetas = np.arccos(np.clip(x_vals, -1.0, 1.0))

# bitstrings for each valid tour
tour_bitstrings = []
for tour in valid_tours:
    bits = []
    for t in range(T_steps):
        bits.extend(int_to_bits(int(tour[t]), n_qubits_step))
    tour_bitstrings.append(bits)
tour_bitstrings = np.array(tour_bitstrings, dtype=int)

# -------------------------
# Block-encoding UA: for each VALID tour basis |s>, apply Ry(2*theta_s) on sig
# IMPORTANT: We will later define Success requiring good=1, so invalid states won't be amplified.
# -------------------------
def apply_UA():
    for bits, theta in zip(tour_bitstrings, thetas):
        apply_controls_for_value(time_wires, bits.tolist())

        # confine RY to the exact matching basis state via MCX-sandwich
        qml.MultiControlledX(wires=time_wires + [sig_wire])
        qml.RY(2.0 * float(theta), wires=sig_wire)
        qml.MultiControlledX(wires=time_wires + [sig_wire])

        undo_controls_for_value(time_wires, bits.tolist())

# -------------------------
# Polynomial utilities for QSVT
# We use a pure even filter: P(x) ~ cosh(t x) / cosh(t), contracted so |P(x)|<=1
# -------------------------
def cheb_power_coeffs_on_minus1_1(fun, deg: int, grid: int = 20000):
    z = np.cos(np.pi * (np.arange(grid) + 0.5) / grid)  # Chebyshev nodes
    y = fun(z)
    cheb_coefs = C.chebfit(z, y, deg)
    pow_coefs = C.Chebyshev(cheb_coefs, domain=[-1, 1]).convert(kind=P.Polynomial).coef
    return np.array(pow_coefs, dtype=float)  # ascending power basis

def max_abs_on_minus1_1(p, grid: int = 200001):
    xx = np.linspace(-1, 1, grid)
    return float(np.max(np.abs(np.polyval(p[::-1], xx))))

def contract_poly(p, safety: float = 1e-6):
    m_ = max_abs_on_minus1_1(p)
    return p / (m_ * (1.0 + safety)), m_

def apply_qsvt_from_angles(angles, apply_UA_fn, work_wire):
    # PennyLane PCPhase applies phase to |0...0> of the wires.
    # Here work_wire is [sig].
    def PC(phi):
        qml.PCPhase(phi, dim=1, wires=work_wire)

    PC(angles[0])
    for j in range(len(angles) - 1):
        if j % 2 == 0:
            apply_UA_fn()
        else:
            qml.adjoint(apply_UA_fn)()
        PC(angles[j + 1])

# QSVT filter parameters
t_filter = 3.0
deg = 40
c = float(np.cosh(abs(t_filter)))

p_even = cheb_power_coeffs_on_minus1_1(lambda x: np.cosh(t_filter * x) / c, deg)
p_even[1::2] = 0.0  # enforce even parity

p_even, _ = contract_poly(p_even, safety=1e-6)
ang_even = qml.poly_to_angles(p_even, "QSVT")

# -------------------------
# Build A := Prep ∘ ValidCompute ∘ QSVT(UA)
# We keep good_wire computed (no intermediate measurements). A† will undo everything as needed.
# -------------------------
def apply_A():
    init_pos_uniform()
    # validity flag
    check_pos_compute()
    # QSVT cost filter (work wire = sig)
    apply_qsvt_from_angles(ang_even, apply_UA, work_wire=[sig_wire])

# -------------------------
# Success reflection: phase flip iff (good=1 AND sig=0)
# Implement with a clean ancilla 'flag_anc' to avoid CCZ assumptions.
# -------------------------
def S_success():
    # condition on sig=0 -> turn it into control-on-1
    qml.PauliX(wires=sig_wire)

    # flag ^= (good AND sig)
    qml.Toffoli(wires=[good_wire, sig_wire, flag_anc])

    # phase flip on flag=1
    qml.PauliZ(wires=flag_anc)

    # uncompute flag
    qml.Toffoli(wires=[good_wire, sig_wire, flag_anc])

    # restore sig
    qml.PauliX(wires=sig_wire)

# -------------------------
# S0 reflection: phase flip on |0...0> of ALL wires used by A and S_success
# (This reflects about the computational |0> state; standard amplitude amplification form.)
# -------------------------
def S0():
    qml.PCPhase(np.pi, dim=1, wires=wires)  # flips |0...0> by -1

# -------------------------
# One amplitude amplification iteration:
# Q = - A S0 A† S_success
# Global minus is irrelevant, so we omit it.
# -------------------------
def AA_step():
    S_success()
    qml.adjoint(apply_A)()
    S0()
    apply_A()

# -------------------------
# Estimate p_success after A to choose K (analytic)
# p_success = Pr(good=1, sig=0)
# -------------------------
@qml.qnode(dev_ana)
def success_probs_after_A():
    apply_A()
    return qml.probs(wires=[good_wire, sig_wire])

probs_gs = success_probs_after_A()
# ordering for probs over 2 qubits is |00>,|01>,|10>,|11> with wire order [good, sig]
# We need good=1 and sig=0 -> |10> -> index 2
p_success = float(probs_gs[2])

if p_success <= 0.0:
    K = 0
else:
    theta = math.asin(min(1.0, math.sqrt(p_success)))
    K = int(math.floor((math.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

print("\nEstimated p_success after A (good=1 & sig=0):", p_success)
print("Chosen AA iterations K =", K)

# -------------------------
# Sampling circuit after AA: sample time_wires only
# -------------------------
@qml.qnode(dev_samp)
def sample_after_AA():
    apply_A()
    for _ in range(K):
        AA_step()
    return qml.sample(wires=time_wires)

samples = sample_after_AA()  # shape (shots, m)

# -------------------------
# Decode + evaluate samples (classically) to report best measured tour
# -------------------------
def decode_path_from_time_bits(sample_row):
    bits = "".join(str(int(b)) for b in sample_row)
    path = []
    for t in range(T_steps):
        chunk = bits[t * n_qubits_step : (t + 1) * n_qubits_step]
        path.append(int(chunk, 2))
    return path

def is_valid_classical(path):
    return sorted(path) == list(range(n - 1))

def classical_cost_norm(path):
    return tour_cost_norm_from_indices(path)

rows = []
best = None  # (cost, path)

for r in range(shots):
    path = decode_path_from_time_bits(samples[r])
    valid = is_valid_classical(path)
    if valid:
        cval = classical_cost_norm(path)
        if best is None or cval < best[0]:
            best = (cval, path)
        rows.append({"path": path, "valid": 1, "cost_norm": cval})
    else:
        rows.append({"path": path, "valid": 0, "cost_norm": np.nan})

df = pd.DataFrame(rows)

valid_rate = float(df["valid"].mean())
print("\nMeasured valid fraction:", valid_rate)
if best is not None:
    print("Best measured path:", best[1], "cost_norm:", best[0])
else:
    print("No valid path measured. Increase shots or adjust (t_filter, deg).")

# show frequent valid paths
valid_df = df[df["valid"] == 1].copy()
if len(valid_df) > 0:
    valid_df["path_str"] = valid_df["path"].astype(str)
    top = valid_df["path_str"].value_counts().head(15).reset_index()
    top.columns = ["path", "count"]
    print("\nTop-15 most frequent valid paths:")
    print(top.to_string(index=False))

out_path = "tsp_valid_qsvt_oneAA_samples.csv"
df.to_csv(out_path, index=False)
print("\nSaved:", out_path)
# %%