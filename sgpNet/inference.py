# coding=utf-8
"""
.. moduleauthor:: Shuhua Gao

This module :mod:`inference` defines the entry point of network inference.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Union
from deap import tools
from .stats_target import get_fitness_values, get_tree_height
from .gputils import NetPrimitiveSet, simplify
from .coevolution import Species, Environment


def _choose_top_genes(importance: pd.Series, k: int, genes: [str]) -> list:
    candidate_indices = []
    for candidate_regulator in importance.sort_values(ascending=False).index[:k]:
        candidate_indices.append(genes.index(candidate_regulator))
    return candidate_indices


def _choose_genes_above_average(importance: pd.Series, genes: [str]) -> list:
    avg = importance.sum() / (len(importance) - 1)  # exclude itself
    candidate_indices = []
    for candidate_regulator in importance.index[importance > avg]:
        candidate_indices.append(genes.index(candidate_regulator))
    return candidate_indices


def infer_Boolean_network(binary_data: Union[pd.DataFrame, str], initial_states: Union[pd.DataFrame, str],
                          genes=None, importance: Union[pd.DataFrame, str]= None,
                          top_k_important=None, self_regulation=False, population_size=100,
                          cxpb=0.6, mutpb=0.5, n_gen_stage1=20, n_gen_stage2=100, controlled_elitism=False,
                          parallel=True, work_dir='.', experiment_name=None, archive_size=5,
                          stable_states=None, w_ssv=0, time_order=None, w_tov=0, verbose=True):
    """

    :param binary_data: pandas DataFrame or str. If str, then it should be a CSV file which can be read to get a
        DataFrame. The column labels should be gene names, while the row labels are certain IDs.
        Each row represents a single cell's state.
    :param initial_states: pandas DataFrame or str. If str, then it should be a CSV file which can be read to get a
        DataFrame. The column labels should be gene names, while the row labels are certain IDs.
        Each row represents a single cell's state. One or more initial states are allowed.
    :param genes: iterable, a list of gene names to be considered for network inference. If `None`, then all genes,
        i.e., the column names of *binary_data*, are considered. Note that a gene name must also be a valid Python
        variable name. Thus, a gene name should not contain characters like '.' and '-'.
    :param importance: pandas DataFrame or str. If str, then it should be a CSV file which can be read to get a
        DataFrame. The column labels and row labels should be same gene names. If `None`, then no feature importance is
        available. Consequently, no feature selection is performed before evolution, which is not recommended.
    :param top_k_important: int, specify the number of most important genes to be considered as potential regulators.
        If None, then genes whose importance is above the average value will be taken as the potential regulators.
    :param self_regulation: bool, whether self regulation is considered.
    :param population_size: int, population size of evolution
    :param cxpb: float, cross-over (mating) probability
    :param mutpb: float, mutation probability
    :param n_gen_stage1: generations in stage 1
    :param n_gen_stage2: generations in stage 2
    :param controlled_elitism: bool, whether to apply controlled elitism for NSGA-II
    :param parallel: bool, whether to parallelize the coevolution on multiple processes
    :param work_dir: str, working directory (to store and read data, etc.), use the current directory by default
    :param experiment_name: str, name of this inference experiment. A folder with this name will be created. If None,
        then a name using current date-time is used by default.
    :param archive_size: int, size of the archive of each species to store non-dominated solutions ever found
    :param stable_states: pandas DataFrame or str. If str, then it should be a CSV file which can be read to get a
        DataFrame. The column labels should be gene names, while the row labels are certain IDs.
        Each row represents a single cell's state. One or more stable states are allowed.
    :param w_ssv: a positive number, penalty weight if a stable state constraint is violated
    :param time_order:
    :param w_tov:
    :param verbose:
    :return:
    """

    # read the csv file if needed
    if isinstance(binary_data, str):
        binary_data = pd.read_csv(binary_data, index_col=0)
    if isinstance(initial_states, str):
        initial_states = pd.read_csv(initial_states, index_col=0)
    if isinstance(importance, str):
        importance = pd.read_csv(importance, index_col=0)
    assert isinstance(binary_data, pd.DataFrame)
    assert isinstance(initial_states, pd.DataFrame)
    assert isinstance(importance, pd.DataFrame) or isinstance(importance, None)
    assert isinstance(top_k_important, int) or top_k_important is None

    if genes is None:
        genes = binary_data.columns.tolist()
    for g in genes:
        assert g.isidentifier(), f"The gene name '{g}' is not a valid Python variable name. Please change it.'"
    binary_data = binary_data[genes]
    initial_states = initial_states[genes]
    if importance is not None:
        importance = importance[genes]

    if stable_states is not None:
        assert w_ssv > 0
        if isinstance(stable_states, str):
            stable_states = pd.read_csv(stable_states)[genes]
    if w_ssv > 0:
        assert stable_states is not None

    # statistics to be inspected
    stats_fitness = tools.Statistics(key=get_fitness_values)
    stats_fitness.register('min', np.min, axis=0)
    stats_fitness.register('max', np.max, axis=0)
    stats_fitness.register('mean', np.mean, axis=0)
    stats_height = tools.Statistics(key=get_tree_height)
    stats_height.register('min', np.min)
    stats_height.register('max', np.max)
    stats_height.register('mean', np.mean)
    mstats = tools.MultiStatistics(fitness=stats_fitness, height=stats_height)

    # build the primitive set for each gene
    psets = []
    for target_index, target_gene in enumerate(genes):
        if importance is None:
            candidate_indices = list(range(len(genes)))
            if not self_regulation:
                candidate_indices.remove(target_index)
        else:
            if top_k_important is None:
                candidate_indices = _choose_genes_above_average(importance[target_gene], genes)
            else:
                candidate_indices = _choose_top_genes(importance[target_gene], top_k_important, genes)
            if self_regulation and target_index not in candidate_indices:
                candidate_indices.append(target_index)
        pset = NetPrimitiveSet(target_index, candidate_indices, genes)
        psets.append(pset)

    # spawn a species for each gene
    species_list = []
    for target_index, target_gene in enumerate(genes):
        s = Species(target_index, population_size, initial_states.values, binary_data.values, psets, archive_size,
                    stable_states=None if stable_states is None else stable_states.values, mstats=mstats)
        species_list.append(s)

    # create the environment
    work_path = Path(work_dir)
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
    experiment_path = work_path / 'experiments' / experiment_name
    env = Environment(species_list, experiment_path)

    # evolve
    print(f'[Experiment: {experiment_name}]')
    print(f"- Begin evolution! \n\tResults can be found in '{experiment_path}' when finished.")
    begin_time = datetime.now()
    k_elites = max(1, int(0.02 * population_size))
    env.coevolve(cxpb, mutpb, n_gen_stage1, n_gen_stage2, k_elites,
                 controlled_elitism, w_ssv=w_ssv, parallel=parallel, verbose=verbose)
    print(f'\n- Evolution finished! \n\t Use time {(datetime.now() - begin_time).seconds} seconds.')
    print(f"\t Results can be found in '{experiment_path}'.")

    # report results
    solutions = []
    found = set()
    for i, s in enumerate(species_list):
        for j, collaboration in enumerate(s.hof_stage2):
            # simplify and remove the repeated solutions
            collaboration_key = ''
            for k, ind in enumerate(collaboration):
                ind.simplified = str(simplify(ind, species_list[k].pset))
                collaboration_key += ind.simplified
            if collaboration_key not in found:
                found.add(collaboration_key)
                solutions.append(collaboration)
    solutions.sort(key=lambda c: c.fitness.values)
    # write the network results into a txt
    txt_path = experiment_path / 'inferred_networks.txt'
    with open(txt_path, 'w') as f:
        for k, collaboration in enumerate(solutions):
            f.write(f'Network #{k: <2d} with fitness: {collaboration.fitness.values}\n', )
            for i, ind in enumerate(collaboration):
                f.write(f'\t{species_list[i].target_gene :<6} <- {ind.simplified}\n')
    if verbose:
        print(f"\n- Boolean network evolution results (only networks of top 5 global fitness reported here)"
              f"\n\t All archived non-dominated solutions during evolution can be found in '{txt_path}'.")
        for k, collaboration in enumerate(solutions[:10]):
            print(f'\nNetwork #{k: <2d} with fitness: {collaboration.fitness.values}', )
            for i, ind in enumerate(collaboration):
                print(f'\t{species_list[i].target_gene :<6} <- {ind.simplified}')
