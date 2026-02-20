import itertools

import numpy as np
import pennylane as qml
import numba

def generate_cost_matrix(n, value_range=(0, 1)):
    cost_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i):
            cost_matrix[i, j] = np.random.uniform(*value_range)
            cost_matrix[j, i] = cost_matrix[i, j]
    return cost_matrix


def generate_all_walks(n, start_node=0):
    a = [i for i in range(n)]
    a.pop(start_node)
    res =  list(itertools.permutations(a))
    for i in range(len(res)):
        res[i] = (start_node,) + res[i] + (start_node,)
    return np.array(res, dtype=int)


def find_all_cost(cost_matrix, walks_array):
    n_walks = walks_array.shape[0]
    n_steps = cost_matrix.shape[0]
    costs = np.empty(n_walks)
    for i in range(n_walks):
        cost = 0
        for j in range(n_steps):
            cost += cost_matrix[walks_array[i, j], walks_array[i, j+1]]
        costs[i] = cost
    return costs


def find_best_walk(all_walks, all_costs):
    best_walk_index = np.argmin(all_costs)
    return all_walks[best_walk_index], all_costs[best_walk_index]


def generate_classical_problem(n):
    cost_matrix = generate_cost_matrix(n)
    all_walks = generate_all_walks(n)
    all_costs = find_all_cost(cost_matrix, all_walks)
    best_walk, best_cost = find_best_walk(all_walks, all_costs)
    return cost_matrix, all_walks, all_costs, best_walk, best_cost