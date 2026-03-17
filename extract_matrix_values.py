# %%
 
import numpy as np
import pandas as pd
import pennylane as qml
import src.classical_funcs as cf
# %%
np.random.seed(42)

n = 5
start_node = n - 1

cost_matrix_raw = cf.generate_cost_matrix(n)
all_walks = cf.generate_all_walks(n, start_node=start_node)
all_costs_raw = cf.find_all_cost(cost_matrix_raw, all_walks)
# %%
