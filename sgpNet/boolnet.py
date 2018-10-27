"""
Utilities about Boolean networks.
"""
import numpy as np
from ortools.graph import pywrapgraph


class BooleanNetwork:
    """
    A Boolean network model for target_gene regulatory networks.
    """
    def __init__(self, update_functions):
        """
        Initialize a Boolean network with the given Boolean update functions for each gene.

        :param update_functions: iterable, each item is a Boolean update function. Requirements: (1) accept a single 1d
            binary array input representing a state vector; (2) return a binary value denoting the new state of a target_gene.
        """
        self._n_genes = len(update_functions)
        self._fs = list(update_functions)

    def async_simulate(self, initial_states):
        """
        Simulate the Boolean network model asynchronously starting from the given initial state.
        :param initial_states: array-like, which contains 0 or 1 for each target_gene
            We may provide one initial state in a 1d array / list, or multiple initial states in a 2d array, where each
            row denotes an initial state.
        :return: set, representing the model state space reachable from the initial state with the asynchronous update
            strategy, where each item is a binary state.
        In an asynchronous update scheme, at most one gene is updated between two consecutive states.
        """
        # use depth-first search-here
        initial_states = np.array(initial_states)
        if initial_states.ndim == 1:    # only one initial state
            assert len(initial_states) == self.n_genes
            initial_states = np.reshape(initial_states, (-1, len(initial_states)))
        else:
            assert initial_states.shape[1] == self.n_genes
        q = list(initial_states)
        model_space = set(tuple(s) for s in initial_states)
        while q:
            old_state = q.pop()
            for i in range(self._n_genes):  # if the ith target_gene is to be updated
                new_state = np.copy(old_state)
                if old_state is None:
                    print('old state is None')
                new_state[i] = self._fs[i](old_state)
                new_state_t = tuple(new_state)  # because numpy array cannot be hashed
                if new_state_t not in model_space:
                    model_space.add(new_state_t)
                    q.append(new_state)
        return model_space

    @property
    def n_genes(self):
        return self._n_genes

    @property
    def update_functions(self):
        return self._fs


def _compute_hamming_distance(model_states: np.ndarray, data_states: np.ndarray):
    """
    Compute the Hamming distance between each model state and each data state.
    :param model_states: array-like, each row is a state
    :param data_states: array-like, each row is a state
    :return: array-like, a distance matrix d, where d[target_index,j] is the Hamming distance between the ith model state and
        the jth data state.
    If one space A has more states, then complement the other space B with virtual states, whose distance from each
    state in B is identical, defined to be the number of genes, target_index.e., number of columns in model_states and data_states.
    """
    # if there is only one state in a space, the input may be 1-d. First turn it into 2d.
    if model_states.ndim == 1:
        model_states = np.reshape(model_states, (1, -1))
    if data_states.ndim == 1:
        data_states = data_states.reshape(1, -1)
    n_genes = model_states.shape[1]
    # Hamming
    d = np.apply_along_axis(lambda s: np.abs(s - data_states).sum(axis=1), axis=1, arr=model_states)
    # complement the smaller dimension with distance n_individuals to get a square matrix
    n_rows, n_columns = d.shape
    if n_rows < n_columns:
        constant_fill = np.full((n_columns - n_rows, n_columns), n_genes)
        d = np.vstack((d, constant_fill))
    elif n_columns < n_rows:
        constant_fill = np.full((n_rows, n_rows - n_columns), n_genes)
        d = np.hstack((d, constant_fill))
    return d


def compute_state_space_distance(model_states, data_states, diff_threshold=None):
    """
    Quantify the difference between model states and data states: each state is a binary vector.

    :param model_states: array-like, each row is a state
    :param data_states: array-like, each row is a state
    :param diff_threshold: if the number of states differ more than this threshold, then no need to compute the exact
        distance. Instead, simply return abs(#data_states - #model_states) * #genes * 2. If set to be None, ignore it.
    :return: a non-negative scalar representing the space distance/dissimilarity

    Model the two state space matching as an assignment problem:
        minimum weight perfect matching in a weighted bipartite graph.
    If one space A has more states, then complement the other space B with virtual states, whose distance from each
    state in B is identical, defined to be the number of genes, target_index.e., number of columns in model_states and
    data_states.
    """
    if not isinstance(data_states, np.ndarray):
        data_states = np.array(list(data_states))
    assert data_states.ndim == 2, 'The data_states must be of 2 dimension.'
    if diff_threshold is not None:
        n_genes = data_states.shape[1]
        model_states = model_states.reshape((-1, n_genes))
        if abs(model_states.shape[0] - data_states.shape[0]) >= diff_threshold:
            return abs(model_states.shape[0] - data_states.shape[0]) * n_genes * 2

    d = _compute_hamming_distance(model_states, data_states)    # distance matrix
    assignment = pywrapgraph.LinearSumAssignment()  # linear assignment problem solver
    # add arc with cost
    m = d.shape[0]
    for i in range(m):
        for j in range(m):
            assignment.AddArcWithCost(i, j, int(d[i, j]))
    if assignment.Solve() == assignment.OPTIMAL:
        return assignment.OptimalCost()
    raise RuntimeError('Failed to solve the assignment problem!')


if __name__ == '__main__':
    def f1(x):
        return x[0] and x[1]
    def f2(x):
        return x[0] or x[2]
    def f3(x):
        return x[3]
    def f4(x):
        return x[0] and (not x[1] or x[2])
    net = BooleanNetwork([f1, f2, f3, f4])
    x0 = [0, 0, 1, 0]
    print(net.async_simulate(x0))
    x1 = [1, 1, 1, 0]
    print(net.async_simulate(x1))
    print(net.async_simulate([x0, x1, [0, 0, 0, 1]]))






