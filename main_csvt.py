# qsvt_tsp_boost.py
# ------------------------------------------------------------
# QSVT-based boosting of best TSP tours (small-n, simulator-friendly)
#
# Idea:
#   - Use the same TSP walk encoding style as your main.py (time registers).
#   - Define a diagonal "quality" signal x(pi) in [-1,1] where x≈1 means good (low cost).
#   - Build a block-encoding UA by applying a controlled Ry rotation on a signal ancilla:
#         Ry(2*theta_pi) on ancilla, with cos(theta_pi)=x(pi).
#     Then the top-left block <0|UA|0> equals diag(x(pi)).
#   - Apply QSVT with a polynomial filter P that increases weight near x=1.
#     Here: exponential-style filter via even/odd Chebyshev fits (like your walks code).
#   - Postselect ancilla=0 (work=0) and read boosted distribution over tours.
#
# Notes:
#   - This is NOT scalable: we implement UA by looping over all valid tours and using
#     multi-controlled operations conditioned on the exact tour bitstring.
#   - Works for n~4..6 in statevector simulation depending on RAM.
# ------------------------------------------------------------

import math
import itertools
import numpy as np
import pandas as pd
import pennylane as qml
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P

# -------------------------
# Classical helpers
# -------------------------
def generate_cost_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.uniform(1.0, 10.0, size=(n, n))
    np.fill_diagonal(M, 0.0)
    return M

def all_valid_tours(n: int, start_node: int):
    nodes = list(range(n))
    nodes.remove(start_node)
    for perm in itertools.permutations(nodes):
        yield perm  # length n-1, permutation of non-start nodes

def tour_cost(cost_matrix: np.ndarray, tour, start_node: int) -> float:
    c = 0.0
    c += cost_matrix[start_node, tour[0]]
    for i in range(len(tour) - 1):
        c += cost_matrix[tour[i], tour[i + 1]]
    c += cost_matrix[tour[-1], start_node]
    return float(c)

# -------------------------
# Bit / indexing utilities (match "time register" style)
# -------------------------
def int_to_bits(x: int, width: int):
    return [(x >> (width - 1 - k)) & 1 for k in range(width)]  # MSB->LSB

def apply_controls_for_value(wires, value_bits):
    # Apply X to flip 0-controls into 1-controls for MultiControlledX patterns
    for w, b in zip(wires, value_bits):
        if b == 0:
            qml.PauliX(wires=w)

def undo_controls_for_value(wires, value_bits):
    for w, b in zip(wires, value_bits):
        if b == 0:
            qml.PauliX(wires=w)

# -------------------------
# Polynomial fitting utilities (as in your walks QSVT code)
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
    m = max_abs_on_minus1_1(p)
    return p / (m * (1.0 + safety)), m

def apply_qsvt_from_angles(angles, apply_UA, work_wires):
    # Phase on |0...0>_work (PennyLane's PCPhase does exactly this)
    def PC(phi):
        qml.PCPhase(phi, dim=1, wires=work_wires)

    PC(angles[0])
    for j in range(len(angles) - 1):
        if j % 2 == 0:
            apply_UA()
        else:
            qml.adjoint(apply_UA)()
        PC(angles[j + 1])

# -------------------------
# Main construction
# -------------------------
def main():
    # ---- TSP instance ----
    n = 5
    start_node = n - 1
    seed = 0

    C_raw = generate_cost_matrix(n, seed=seed)
    tours = list(all_valid_tours(n, start_node))
    costs = np.array([tour_cost(C_raw, t, start_node) for t in tours], dtype=float)

    # Normalize cost -> [0,1]
    Cmax = float(np.max(costs))
    costs_norm = costs / Cmax

    # Define quality x in [-1,1], where x=1 is best (cost=0), x=-1 is worst (cost=1)
    # You can tweak this mapping; it just must stay in [-1,1].
    x_vals = 1.0 - 2.0 * costs_norm

    # ---- Encoding parameters (time registers for n-1 nodes) ----
    T_steps = n - 1                      # number of positions in the permutation
    b = int(np.ceil(np.log2(n - 1)))      # bits per position
    time_wires = [f"t{t}_q{q}" for t in range(T_steps) for q in range(b)]
    def pos_wires_at_t(t):
        return [f"t{t}_q{q}" for q in range(b)]

    # Signal ancilla (work space for block-encoding + QSVT phases)
    sig = "sig"
    # Optional: extra ancilla for building multi-controlled rotations if needed
    # (not used here; MultiControlledX work_wires can be given if your backend needs it)
    wires = time_wires + [sig]

    dev = qml.device("lightning.qubit", wires=wires)

    # ---- Build a lookup: tour -> bit pattern over time_wires ----
    # We represent each non-start node by an integer 0..(n-2). Fix an enumeration:
    nonstart = [v for v in range(n) if v != start_node]
    idx_of_node = {v: i for i, v in enumerate(nonstart)}  # node -> 0..n-2

    # For each tour, compute the concatenated bits over all time steps
    tour_bitstrings = []
    for tour in tours:
        bits = []
        for t in range(T_steps):
            val = idx_of_node[tour[t]]          # in 0..n-2
            bits.extend(int_to_bits(val, b))
        tour_bitstrings.append(bits)
    tour_bitstrings = np.array(tour_bitstrings, dtype=int)

    # ---- Block-encoding unitary UA: controlled Ry on sig with cos(theta)=x(tour) ----
    # For each valid tour basis state |tour>, apply Ry(2*theta_tour) on sig, where theta=arccos(x).
    thetas = np.arccos(np.clip(x_vals, -1.0, 1.0))  # theta in [0,pi]

    def apply_UA():
        # This UA is diagonal-on-system via "select-Ry" conditioned on exact tour bitstrings.
        # It acts nontrivially ONLY on valid tours; on other basis states, it does nothing.
        #
        # WARNING: exponential loop over (n-1)! tours.
        for bits, theta in zip(tour_bitstrings, thetas):
            # Turn exact match on time_wires into all-ones controls
            apply_controls_for_value(time_wires, bits.tolist())

            # Controlled rotation on sig, conditioned on all time_wires being |1...1>
            # Implement as: MCX(time_wires -> sig) ; RY(2*theta) on sig ; MCX back
            # This "toggles" sig only on the matching basis; the sandwich confines the rotation.
            qml.MultiControlledX(wires=time_wires + [sig])
            qml.RY(2.0 * float(theta), wires=sig)
            qml.MultiControlledX(wires=time_wires + [sig])

            undo_controls_for_value(time_wires, bits.tolist())

    # ---- Initial state: uniform over ALL time_wires (then only valid tours matter after postselect) ----
    def prep_uniform():
        for w in time_wires:
            qml.Hadamard(wires=w)
        # sig starts at |0>

    # ---- Choose a QSVT polynomial filter P(x) that boosts near x=1 ----
    # Exponential-style shaping: exp(t x) with t>0 boosts larger x, then rescale to stay bounded.
    # Like your walks code: implement cosh/sinh pieces and combine via postselected LCU (0.5*(even+odd)).
    t = +3.0          # larger -> sharper preference for high x (low cost)
    deg = 40          # Chebyshev fit degree (increase if you want sharper but more expensive)
    c = np.cosh(abs(t))

    p_even = cheb_power_coeffs_on_minus1_1(lambda x: np.cosh(t * x) / c, deg)
    p_odd  = cheb_power_coeffs_on_minus1_1(lambda x: np.sinh(t * x) / c, deg)

    # Clean parity leakage
    p_even[1::2] = 0.0
    p_odd[0::2]  = 0.0

    # Contract so |P(x)|<=1 numerically, otherwise qml.poly_to_angles may fail
    p_even, scale_even = contract_poly(p_even, safety=1e-6)
    p_odd,  scale_odd  = contract_poly(p_odd,  safety=1e-6)

    ang_even = qml.poly_to_angles(p_even, "QSVT")
    ang_odd  = qml.poly_to_angles(p_odd,  "QSVT")

    work_wires = [sig]  # QSVT phase is on |0> of sig

    @qml.qnode(dev)
    def qsvt_state_even():
        prep_uniform()
        apply_qsvt_from_angles(ang_even, apply_UA, work_wires)
        return qml.state()

    @qml.qnode(dev)
    def qsvt_state_odd():
        prep_uniform()
        apply_qsvt_from_angles(ang_odd, apply_UA, work_wires)
        return qml.state()

    print("Simulating QSVT-even statevector ...")
    st_even = qsvt_state_even()
    print("Simulating QSVT-odd  statevector ...")
    st_odd = qsvt_state_odd()

    # ---- Read boosted distribution over VALID tours by postselecting sig=0 ----
    # In this encoding, tour basis states live on time_wires; sig is the last wire.
    # We'll compute amplitude for each tour AND sig=0, combine even/odd as 0.5*(a_even+a_odd),
    # and then renormalize over tours (postselection).
    #
    # Note: PennyLane statevector indexing depends on wire order; lightning.qubit uses
    # the order provided in device wires. Here: [time_wires..., sig].
    #
    def index_from_bits(system_bits, sig_bit):
        full_bits = system_bits + [sig_bit]   # MSB->LSB relative to wires list
        idx = 0
        for b_ in full_bits:
            idx = (idx << 1) | int(b_)
        return idx

    rows = []
    joint_sum = 0.0

    # Also compute classical rank for comparison
    order = np.argsort(costs)  # ascending cost
    best_cost = float(costs[order[0]])
    print(f"Best classical tour cost = {best_cost:.6f} (n={n}, start={start_node})")

    for k, (tour, bits, cost_val, x) in enumerate(zip(tours, tour_bitstrings, costs, x_vals)):
        sys_bits = bits.tolist()
        idx0 = index_from_bits(sys_bits, 0)  # sig=0
        a_even = st_even[idx0]
        a_odd  = st_odd[idx0]
        a_exp  = 0.5 * (a_even + a_odd)      # postselected LCU combination in amplitude

        p_joint = float(np.abs(a_exp) ** 2)
        joint_sum += p_joint

        rows.append({
            "tour": str(tour),
            "cost": float(cost_val),
            "cost_norm": float(cost_val / Cmax),
            "x=1-2*cost_norm": float(x),
            "amp_real": float(np.real(a_exp)),
            "amp_imag": float(np.imag(a_exp)),
            "p_joint(sig=0)": p_joint,
        })

    df = pd.DataFrame(rows).sort_values("p_joint(sig=0)", ascending=False).reset_index(drop=True)

    # Renormalize -> conditional distribution given postselection sig=0
    df["p_cond(tour | sig=0)"] = df["p_joint(sig=0)"] / (joint_sum if joint_sum > 0 else 1.0)

    print("\nTop-10 tours by boosted conditional probability:")
    print(df[["tour","cost","p_cond(tour | sig=0)"]].head(10).to_string(index=False))

    print(f"\nTotal postselection probability Pr(sig=0) over valid tours (joint_sum) = {joint_sum:.6e}")
    print("\n(If joint_sum is very small, you are using a very sharp filter / high degree / t, or n is too large.)")

    # Save CSV like your workflow
    out = "tsp_qsvt_boost_distribution.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote: {out}")

if __name__ == "__main__":
    main()
