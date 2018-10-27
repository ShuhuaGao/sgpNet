"""
Define some `getters` for statistics computing. To facilitate multi-processing parallelism, these functions must be
defined in the top level.
"""


def get_fitness_values(ind):
    return ind.fitness.values


def get_tree_height(ind):
    return ind.height
