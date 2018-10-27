"""
Cooperative coevolution framework.
"""
import numpy as np
from deap import gp
from deap import tools, base
from pathlib import Path
import random
from deap import algorithms
from .boolnet import compute_state_space_distance
from .cboolnet import BooleanNetwork
from .gputils import NetPrimitiveSet, Archive
import multiprocessing as mp


class FitnessMin(base.Fitness):
    """
    Represent a fitness objective to be minimized.
    """
    weights = (-1.0,)


class Individual(gp.PrimitiveTree):
    """
    An individual (chromosome) representing a syntax tree in GP.
    """

    def __init__(self, content):
        """
        Create an individual with the given content and pset.

        :param content: iterable
        :param pset: gputils.NetPrimitiveSet
        """
        super().__init__(content)
        self.fitness = FitnessMin()
        self.n_stable_state_violations = 0
        self.n_time_order_violations = 0
        self.n_regulators = 0
        self.compiled = None


def compile_with_primitives(individual: Individual, pset: NetPrimitiveSet):
    """
    Compile an individual into a Boolean update function using the given primitive set.

    :param individual: Individual
    :param pset: primitive set
    :return: a Lambda representing the Boolean update function
    """
    raw_lambda = gp.compile(individual, pset)

    # we need a Boolean update function, which accepts all the genes as inputs
    # however, the lambda compiled from individual only accepts its candidate regulators
    return lambda s: raw_lambda(*[s[i] for i in pset.candidate_indices])


class Collaboration:
    """
    A cooperation of individuals to form a complete solution.
    """

    def __init__(self, species_index, gen, individuals, fit=None):
        self.species_index = species_index
        self.gen = gen
        self.individuals = individuals
        self.fitness = FitnessMin()
        if fit is not None:
            self.fitness.values = fit

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, item):
        return self.individuals[item]


class Species:
    """"
    Represent a _species_list in the coevolution framework.
    """

    def __init__(self, index, n, initial_states, data_states, primitive_sets, archive_size=10,
                 stable_states=None, mstats=None):
        """
        Initialize a species composed of n_individuals individuals.

        :param index: int, index of this species among all the species
        :param n: int, number of individuals in this species
        :param initial_states: 2d list, one or more initial states denoting the starting point of gene network
        :param data_states: 2d list, all the binary states observed in the experiment
        :param primitive_sets: list of gputils.NetPrimitiveSet,
            primitive sets for all the species which is used when cooperating
        :param archive_size: int, size of the archive for non-dominated individuals
        :param stable_states: 2d list, once the network reaches a stable state, it should not leave it
        :param mstats: deap.tools.Statistics or deap.tools.MultiStatistics, statistics to be monitored during evolution
        """

        self.index = index
        self.n_individuals = n
        self._experiment_path = None
        self.log_file = None

        self.data_states = {tuple(s) for s in data_states}  # facilitate data/model space matching
        self.initial_states = initial_states
        self.stable_states = stable_states
        self.primitive_sets = primitive_sets

        self.pset = self.primitive_sets[index]
        self.toolbox = self._init_toolbox()
        self.population = None

        self.collaboration_pool_size = 3

        self.gen = 0
        self.stage = 1
        self.n_gen_stage1 = 50  # total number of generations in stage1
        self._init_evolution()

        self.stage = 1
        self.best_collaboration_history = {}

        self.hof_stage1 = tools.HallOfFame(archive_size)
        self.hof_stage2 = Archive(archive_size)
        self.best_global_fitness = None
        self.best_local_fitness = None
        self._n_stagnation = 0

        self.is_single_objective = False  # if true, then only the global fitness is used

        self.logbook = tools.Logbook()
        self.mstats = mstats
        self.logbook.header = ['index', 'stage', 'gen'] + mstats.fields if mstats is not None else []

# region evolution initialization
    @staticmethod
    def _mutate(individual, expr, pset):
        r = random.random()
        if r < 0.9:
            return gp.mutUniform(individual, expr, pset)
        else:
            return gp.mutNodeReplacement(individual, pset)

    def _init_toolbox(self):
        """
        Initialize a toolbox

        :return: base.Toolbox
        """
        toolbox = base.Toolbox()
        toolbox.register('expr', gp.genHalfAndHalf, min_=0, max_=2, pset=self.pset)
        toolbox.register('individual', tools.initIterate, Individual, toolbox.expr)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register('compile', compile_with_primitives, pset=self.pset)
        # genetic variations
        toolbox.register('mate', gp.cxOnePoint)
        toolbox.register('expr_mut', gp.genHalfAndHalf, min_=0, max_=3)  # expression for mutation
        toolbox.register('mutate', self._mutate, expr=toolbox.expr_mut, pset=self.pset)
        return toolbox

    def _init_evolution(self):
        """
        Prepare and initialize the evolution process.
        """
        self.population = self.toolbox.population(self.n_individuals)
        self.gen = 0
        self.stage = 1
# endregion

    def _compile(self, population):
        """
        Compile each individual in *population* and assign the result to a new `compiled` attribute

        :param population:
        """
        for ind in population:
            ind.compiled = self.toolbox.compile(ind)


# region penalty
    @staticmethod
    def _count_num_regulators(population):
        """
        Get the number of distinct input variables in each individual of *population*.
        Assign it to the 'n_regulators'attribute.

        :param population: a list of individuals
        """
        for individual in population:
            input_variables = set()
            for node in individual:
                if isinstance(node, gp.Terminal):  # the terminal can only be an input since no constants exist
                    input_variables.add(node.name)
            individual.n_regulators = len(input_variables)
# endregion

# region local fitness
    def _evaluate_false_states(self, individual):
        """
        Count the number of false states induced by this *individual*.

        :param individual:  an individual of current species list
        :return: a tuple of one element, the number of false states
        """
        update_function = individual.compiled
        false_states_count = 0
        for s in self.data_states:
            s = np.copy(s)
            s[self.index] = update_function(s)
            if tuple(s) not in self.data_states:
                false_states_count += 1
        return false_states_count

    def _count_stable_state_violations(self, population):
        """
        The network should not leave any stable state. Count how many stable states are violated for each individual.

        :param population: a population
        :return: int
        """
        for ind in population:
            update_function = ind.compiled
            count = 0
            for ss in self.stable_states:
                old = ss[self.index]
                new = update_function(ss)
                if old != new:
                    count += 1
            ind.n_stable_state_violations = count

    def _evaluate_local(self, individual, w_ssv=None):
        """
        Evaluate the local fitness, which may include two parts: number of false states and violations of stable state
        constraints.

        :param individual:
        :param w_ssv: float, weight for constraint violations.
            If w is None, then the stable state constraints are not considered.
        :return: float
        """
        n_false_states = self._evaluate_false_states(individual)
        fit = n_false_states
        if w_ssv is not None:
            fit += w_ssv * individual.n_stable_state_violations
        return fit
# endregion

# region global fitness
    def _evaluate_space_distance(self, individual, collaboration_representatives, w_ssv=None):
        """
        Evaluate the distance between the model space and the data space.

        :param individual: an individual of current _species_list
        :param collaboration_representatives: list of list, ith row contains the representatives from the ith species
        :param w_ssv: float, weight for penalty of stable state constraint violations.
            If None, then ignore these constraints.
        :return: tuple, the minimal distance of the best collaboration, the best collaboration
        """
        self_func = individual.compiled

        min_dist = float('inf')
        best_team = None  # the best cooperation for this individual
        for _ in range(self.collaboration_pool_size):
            # construct a collaboration
            update_functions = [None] * self.n_species
            team = [None] * self.n_species
            for i in range(len(update_functions)):
                if i == self.index:
                    update_functions[i] = self_func
                    team[i] = individual
                else:
                    j = random.randint(0, len(collaboration_representatives[i]) - 1)
                    team[i] = collaboration_representatives[i][j]
                    update_functions[i] = team[i].compiled
            # evaluate this collaboration
            bnet = BooleanNetwork(update_functions)
            model_states = bnet.async_simulate(self.initial_states)
            # TODO: optimization
            model_states = np.array(model_states)
            data_states = np.array(list(self.data_states))
            dist = compute_state_space_distance(model_states, data_states)
            # penalty of stable state violations
            if w_ssv is not None:
                dist += w_ssv * sum(ind.n_stable_state_violations for ind in team)
            # a better one?
            if dist < min_dist:
                min_dist = dist
                best_team = team
        return min_dist, Collaboration(self.index, self.gen, best_team)

    def _evaluate_space_mismatch(self, individual, collaboration_representatives, w_ssv=None):
        """
        Evaluate the mismatch between the model space and the data space, i.e., number of states that only appear in
        one space.

        :param individual: an individual of current _species_list (compiled or not)
        :param collaboration_representatives: list of list, ith row a list of representatives from ith species
        :param w_ssv: [float, float] weight for penalty of stable state constraint violations, only the 2nd weight is used.
            If None, then ignore these constraints.
        :return: tuple, the minimal distance of the best collaboration, the best collaboration
        """
        self_func = individual.compiled

        min_dist = float('inf')
        best_team = None  # the best cooperation for this individual
        for _ in range(self.collaboration_pool_size):
            # construct a collaboration
            update_functions = [None] * self.n_species
            team = [None] * self.n_species
            for i in range(len(update_functions)):
                if i == self.index:
                    update_functions[i] = self_func
                    team[i] = individual
                else:
                    j = random.randint(0, len(collaboration_representatives[i]) - 1)
                    team[i] = collaboration_representatives[i][j]
                    update_functions[i] = team[i].compiled

            # evaluate this collaboration
            bnet = BooleanNetwork(update_functions)
            model_states = bnet.async_simulate(self.initial_states)
            model_states = {tuple(s) for s in model_states}
            dist = len(model_states ^ self.data_states)
            # penalty of stable state violations
            if self.stable_states is not None:
                dist += w_ssv * sum(ind.n_stable_state_violations for ind in team)
            # a better one?
            if dist < min_dist:
                min_dist = dist
                best_team = team
        return min_dist, Collaboration(self.index, self.gen, best_team)

    def _evaluate_global(self, individual, collaboration_representatives, method='mismatch', w_ssv=None):
        """
        Evaluate the global fitness of an *individual* through collaboration.

        :param individual:
        :param collaboration_representatives:
        :param method: 'mismatch' or 'distance', how to measure the difference between the model space and data space
        :param w_ssv: penalty weight of stable state violations. If None, then no penalty
        :return: tuple of two elements, the global fitness and the collaboration leading to this fitness
        """
        if method == 'mismatch':
            fit, collab = self._evaluate_space_mismatch(individual, collaboration_representatives, w_ssv)
        elif method == 'distance':
            fit, collab = self._evaluate_space_distance(individual, collaboration_representatives, w_ssv)
        else:
            raise RuntimeError("The method argument only supports 'mismatch' or 'distance'.")
        return fit, collab
# endregion

# region evaluation for two stages
    def _evaluate_stage1(self, individual, w_ssv=None):
        """
        Evaluate stage 1, which is single-objective and only considers the local fitness and constraints if any.

        :return: a tuple of one item
        """
        return self._evaluate_local(individual, w_ssv=w_ssv),

    def _evaluate_stage2(self, individual, collaboration_representatives, method='mismatch', w_ssv=None):
        """
        Evaluate multiple fitnesses for multi-objective evolution in stage 2. The three fitnesses are
        (global_fitness, local_fitness, model_complexity).

        :param individual: an individual
        :param collaboration_representatives:
        :param method: 'mismatch' or 'distance', how to measure the difference between the model space and data space
        :param w_ssv: penalty weight of stable state violations. If None, then no penalty
        :return: Collaboration, whose fitness is also the fitness of the given *individual*
        """
        # local fitness
        l_fit = self._evaluate_local(individual, w_ssv)
        # global fitness
        g_fit, collab = self._evaluate_global(individual, collaboration_representatives, method, w_ssv)
        # complexity penalty
        penalty = sum(ind.n_regulators for ind in collab)
        # three objectives in total
        collab.fitness.values = (g_fit, l_fit, penalty)
        return collab
# endregion

    def _log_to_file(self):
        with open(self.log_file, 'a') as f:
            f.write(self.logbook.stream + '\n')

    def _update_best_collaboration_history(self, best_collaborations):
        best = max(best_collaborations, key=lambda cb: cb.fitness)  # lexicographic ordering
        self.best_collaboration_history[best.gen] = best

# region evolution
    def evolve_stage1(self, cxpb, mutpb, n_gen=50, k_elites=1, w_ssv=None):
        """
        Perform stage 1 evolution, where only the local fitness (number of false states) matters.
        In this stage, each gene's regulation function is evolved separately and not cooperation is needed.

        :param cxpb: cross-over probability
        :param mutpb: mutation probability
        :param n_gen: number of generations
        :param k_elites: number of elites
        :param w_ssv: penalty weight of stable state violations. If None, then no penalty.
        :return:
        """
        self.stage = 1
        self.n_gen_stage1 = n_gen
        if n_gen == 0:
            return
        # normal single-objective GP
        self.toolbox.register('select', tools.selTournament, tournsize=3)
        self.toolbox.register('evaluate', self._evaluate_stage1, w_ssv=w_ssv)
        FitnessMin.weights = (-1.0,)

        for gen in range(n_gen + 1):
            self.gen = gen
            # individuals that have been changed due to mutation or crossover
            invalid_pop = [ind for ind in self.population if not ind.fitness.valid]
            self._compile(invalid_pop)
            if self.stable_states is not None:
                self._count_stable_state_violations(invalid_pop)
            self._count_num_regulators(invalid_pop)
            # Evaluate the individuals with an invalid fitness
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_pop)
            for ind, fit in zip(invalid_pop, fitnesses):
                ind.fitness.values = fit

            self.hof_stage1.update(self.population)
            self.best_local_fitness = max(self.population, key=lambda ind: ind.fitness).fitness.values[0]
            # log
            if hasattr(self, 'logbook'):
                record = self.mstats.compile(self.population)
                self.logbook.record(index=self.index, stage=self.stage, gen=self.gen, **record)
                self._log_to_file()

            if gen == n_gen:
                break
            # selection and variation to produce the next generation
            elites = tools.selBest(self.population, k_elites)
            parents = self.toolbox.select(self.population, len(self.population) - k_elites)
            offspring = algorithms.varAnd(parents, toolbox=self.toolbox, cxpb=cxpb, mutpb=mutpb)
            self.population = elites + offspring

    def _prepare_stage2(self, w_ssv):
        """
        Prepare NSGA-II. Mainly to assign the crowding distance for the first time.
        """
        # individuals that have been changed due to mutation or crossover
        invalid_pop = [ind for ind in self.population if not ind.fitness.valid]
        self._compile(invalid_pop)
        if self.stable_states is not None:
            self._count_stable_state_violations(invalid_pop)
        self._count_num_regulators(self.population)
        self.stage = 2
        # three objective GP
        FitnessMin.weights = (-1.0, -1.0, -1.0)
        self.toolbox.register('select', tools.selTournamentDCD)
        self.toolbox.register('mutate', self._mutate, expr=self.toolbox.expr_mut, pset=self.pset)
        # evaluate: each individual has its own best collaboration from the collaboration pool
        best_collaborations = list(self.toolbox.map(self.toolbox.evaluate, self.population))
        for ind, best_collaboration in zip(self.population, best_collaborations):
            ind.fitness.values = best_collaboration.fitness.values
        # Just assign the crowding distance for the first time, `fitness.crowding_dist`, which is required in selDCD
        self.population = tools.selNSGA2(self.population, len(self.population))
        self.hof_stage2.update(best_collaborations)
        self._update_best_collaboration_history(best_collaborations)
        self.best_global_fitness = max(self.population, key=lambda ind: ind.fitness).fitness.values[0]

    def evolve_once_stage2(self, collaboration_representatives, cxpb, mutpb,
                           controlled_elitism=True, method='mismatch', w_ssv=None):
        """
        Perform one iteration of evolution in stage 2.

        :param collaboration_representatives: list of list, each row includes one representative from each species
            (including this species itself)
        :param cxpb: crossover probability
        :param mutpb: mutation probability
        :param controlled_elitism: bool, whether to apply controlled elitism for NSGA-II, True by default
        :param method: 'mismatch' or 'distance', how to measure the difference between the model space and data space
        :param w_ssv: float, weight for constraint violations.
            If w is None, then the stable state constraints are not considered.
        :return: representative of this generation
        """
        # because Lambda cannot be pickled, a representative may have not been compiled yet in parallel processing
        for i, reps in enumerate(collaboration_representatives):  # representatives from the ith species
            pset = self.primitive_sets[i]
            for ind in reps:
                if ind.compiled is None:
                    ind.compiled = compile_with_primitives(ind, pset)

        self.toolbox.register('evaluate', self._evaluate_stage2,
                              collaboration_representatives=collaboration_representatives, method=method, w_ssv=w_ssv)

        if self.gen == self.n_gen_stage1:  # just switch into stage 2: do some initialization
            self._prepare_stage2(w_ssv)
        self.gen += 1

        # selection: selTournamentDCD, considering both the Pareto rank and the crowding distance
        parents = self.toolbox.select(self.population, len(self.population))
        # variation: adaptive mutation rate to escape a local minimum
        mutpb = min(0.8, (1 + self._n_stagnation / 50) * mutpb)
        offspring = algorithms.varAnd(parents, self.toolbox, cxpb, mutpb)
        if self._n_stagnation > 5:
            i = random.randint(0, len(offspring) - 1)
            offspring[i] = self.toolbox.individual()

        # measure the characteristics of varied individuals due to crossover or mutation
        invalid_pop = [ind for ind in offspring if not ind.fitness.valid]
        self._compile(invalid_pop)  # set attr 'compiled'
        if self.stable_states is not None:
            self._count_stable_state_violations(invalid_pop)    # set attr 'n_stable_state_violations'
        self._count_num_regulators(invalid_pop)     # set attr 'n_regulators'
        # evaluate all fitness: even an individual is not varied, the representatives have been changed
        best_collaborations = list(self.toolbox.map(self.toolbox.evaluate, offspring))

        for ind, best_collaboration in zip(offspring, best_collaborations):
            ind.fitness.values = best_collaboration.fitness.values
        self.hof_stage2.update(best_collaborations)
        self._update_best_collaboration_history(best_collaborations)

        # select N from 2N individuals assembled from parents and offspring
        if controlled_elitism:
            self.population = tools.selNSGA2ControlledElitism(self.population + offspring, len(self.population), r=0.5)
        else:
            self.population = tools.selNSGA2(self.population + offspring, len(self.population))
        # log
        if hasattr(self, 'logbook'):
            record = self.mstats.compile(self.population)
            self.logbook.record(index=self.index, stage=self.stage, gen=self.gen, **record)
            self._log_to_file()
        bgf = max(self.population, key=lambda ind: ind.fitness).fitness.values[0]
        if self.best_global_fitness - bgf < 1:
            self._n_stagnation += 1
        else:
            self._n_stagnation = 0
        self.best_global_fitness = bgf


# endregion

# region properties
    def __str__(self):
        return f'Species {self.species_index} @ gen {self.gen}'

    @property
    def target_gene(self):
        return self.pset.genes[self.index]

    @property
    def experiment_path(self):
        return self._experiment_path

    @experiment_path.setter
    def experiment_path(self, val):
        self._experiment_path = val
        if val is not None:
            self.log_file = self.experiment_path / 'log' / f'species_{self.index}_statistics.log'
            if self.log_file.exists():
                self.log_file.unlink()

    @property
    def n_species(self):
        """
        Get the total number of _species_list, i.e., the total number of genes in the network.
        """
        return len(self.primitive_sets)

    @property
    def representatives(self):
        """
        Get representatives from this _species_list.
        Default: best + random

        :return: a list of representatives, which are copies of certain individuals
        """
        try:
            best_ind = max(self.population, key=lambda ind: ind.fitness)
        except:  # if the population are not evaluated yet, i.e., fitness is still invalid
            best_ind = random.choice(self.population)
        random_ind = random.choices(self.population, k=1)
        random_ind.append(best_ind)
        return [self.toolbox.clone(ind) for ind in random_ind]
# endregion


class Environment:
    """
    Represents an environment for coevolution.
    """

    def __init__(self, species_list, experiment_path: Path = None):
        """
        Construct an environment.
        :param species_list: iterable, all the _species_list to be coevolved
        :param experiment_path: pathlib.Path, denoting the experiment directory. If None, no logging is performed.
        """
        self._species_list = sorted(species_list, key=lambda s: s.index)
        self.logbook = tools.Logbook()
        self._species_symbol = 'species' if len(species_list) <= 12 else 's'
        self.logbook.header = ['stage', 'gen'] + [f'{self._species_symbol}{i}' for i in range(len(species_list))]
        self.gen = 0
        self._verbose = False
        for s in species_list:
            s.experiment_path = experiment_path
        # configure the experiment directory
        if experiment_path is not None:
            experiment_path.mkdir(exist_ok=True, parents=True)
            (experiment_path / 'log').mkdir(exist_ok=True)
            (experiment_path / 'pickle').mkdir(exist_ok=True)
            self.log_file = experiment_path / 'log/environment_global_fitness.log'
            if self.log_file.exists():
                self.log_file.unlink()

    def _print(self, *args):
        if self._verbose:
            print(*args)

    def _coevolve_serial(self, cxpb, mutpb, n_gen_stage1, n_gen_stage2, k_elites,
                         controlled_elitism=True, method='mismatch', w_ssv=None):
        """
        Run coevolution.
        :param cxpb: crossover probability
        :param mutpb: mutation probability
        :param n_gen_stage1: generations in stage 1
        :param n_gen_stage2: generations in stage 2
        :param k_elites: number of elites
        :return: list, representatives of each _species_list after evolution finishes
        """
        self._print('\n- Stage 1: to minimize the local fitness objective.')
        self._print(f'\t (Number of generations = {n_gen_stage1})')
        # stage 1: SO
        for i, s in enumerate(self.species_list):
            s.evolve_stage1(cxpb, mutpb, n_gen_stage1, k_elites, w_ssv)
            self._print(f'- Species {i : <2d} finished. '
                        f'Best local fitness in the last generation is {s.best_local_fitness}.')

        self._print('\n- Stage 2: to minimize (global_fitness, local_fitness, model_complexity) objectives')
        self._print(f'\t (Number of generations = {n_gen_stage2})')
        self._print('- The best global fitness due to cooperation of each species in each generation is listed below.')
        # stage 2: MO
        for gen in range(n_gen_stage1 + 1, n_gen_stage1 + n_gen_stage2 + 2):
            # fetch the representatives from each _species_list
            all_representatives = [s.representatives for s in self.species_list]
            for s in self.species_list:
                s.evolve_once_stage2(all_representatives, cxpb, mutpb, controlled_elitism, method, w_ssv)

            # statistics: get the best collaboration found so far in each _species_list
            self.logbook.record(stage=2, gen=gen,
                                **{f'{self._species_symbol}{i}': s.best_global_fitness
                                   for i, s in enumerate(self.species_list)})
            stream = self.logbook.stream
            self._print(stream)

        if self.log_file is not None:
            self._log_to_file(self.logbook)
        return self.logbook

    def _coevolve_one_species_parallel(self, s, cxpb, mutpb, n_gen_stage1, n_gen_stage2, k_elites, conn,
                                       controlled_elitism=True, method='mismatch', w_ssv=None):
        """
        Evolve one species `s` on an independent process.

        The Connection object `conn` created by multiprocessing.pipe is used to transfer messages among these processes,
        while it can also synchronize the evolution pace of these species.
        """
        # stage 1
        # print(f'[{s.index :<2d}] Start stage 1 on process ', mp.current_process().pid)
        s.evolve_stage1(cxpb, mutpb, n_gen_stage1, k_elites, w_ssv)
        self._print(f'- Species {s.index : <2d} finished. '
                    f'Best local fitness in the last generation is {s.best_local_fitness}.')

        # stage 2
        # print(f'[{s.index :<2d}] Start stage 2 on process ', mp.current_process().pid)
        for gen in range(n_gen_stage1 + 1, n_gen_stage1 + n_gen_stage2 + 2):
            # send own representatives to the master process,
            representatives = s.representatives
            for rep in representatives:
                rep.compiled = None    # a lambda cannot be pickled
            conn.send(representatives)
            # and fetch all the representatives from all species from the master process
            collaboration_representatives = conn.recv()
            s.evolve_once_stage2(collaboration_representatives, cxpb, mutpb, controlled_elitism, method, w_ssv)
            conn.send(s.best_global_fitness)

        # finally, send the final population and the HOF in stage 2 to the parent process
        for ind in s.population:
            ind.compiled = None
        for collab in s.hof_stage2:
            for ind in collab:
                ind.compiled = None
        conn.send((s.population, s.hof_stage2, s.logbook))

    def _coevolve_parallel(self, cxpb, mutpb, n_gen_stage1, n_gen_stage2, k_elites,
                           controlled_elitism=True, method='mismatch', w_ssv=None):
        """
        Evolve the species/sub-population in a cooperative coevolution architecture in a parallel manner by
        deploying each species to a separate process.
        """
        n_processes = len(self.species_list)
        # n pipes, each between one child process (for one species) and the current parent process
        pipes = [mp.Pipe() for _ in range(n_processes)]
        parent_connections = [pipe[0] for pipe in pipes]
        child_connections = [pipe[1] for pipe in pipes]

        self._print('\n- Stage 1: to minimize the local fitness objective.')
        self._print(f'\t (Number of generations = {n_gen_stage1})')

        processes = [mp.Process(target=self._coevolve_one_species_parallel,
                                args=(self.species_list[i], cxpb, mutpb, n_gen_stage1, n_gen_stage2, k_elites,
                                      child_connections[i], controlled_elitism, method, w_ssv)) for i in range(n_processes)]
        for p in processes:
            p.start()

        # master process: receive all representatives from each species, pack them together, and send the pack to each
        # slave process where each species resides
        for gen in range(n_gen_stage1 + 1, n_gen_stage1 + n_gen_stage2 + 2):
            all_representatives = [conn.recv() for conn in parent_connections]
            if gen == n_gen_stage1 + 1:
                self._print('\n- Stage 2: to minimize (global_fitness, local_fitness, model_complexity) objectives')
                self._print(f'\t (Number of generations = {n_gen_stage2})')
                self._print(
                    '- The best global fitness due to cooperation of each species in each generation is listed below.')
            for conn in parent_connections:
                conn.send(all_representatives)
            # statistics: get the best collaboration found so far in each species
            best_global_fitnesses = [conn.recv() for conn in parent_connections]
            self.logbook.record(stage=2, gen=gen,
                                **{f'{self._species_symbol}{i}': g_fit
                                   for i, g_fit in enumerate(best_global_fitnesses)})
            stream = self.logbook.stream
            self._print(stream)

        # evolution finished: get the evolved species from other processes
        for conn, s in zip(parent_connections, self.species_list):
            s.population, s.hof_stage2, s.logbook = conn.recv()
        FitnessMin.weights = (-1, -1, -1)  # class attribute cannot be picked and sent inter-process

        for p in processes:
            p.join()

        if self.log_file is not None:
            self._log_to_file(self.logbook)
        return self.logbook

    def coevolve(self, cxpb, mutpb, n_gen_stage1=50, n_gen_stage2=50, k_elites=1,
                 controlled_elitism=True, method='mismatch', w_ssv=None, parallel=True, verbose=__debug__):
        """
        Run coevolution.

        :param cxpb: crossover probability
        :param mutpb: mutation probability
        :param n_gen_stage1: generations in stage 1
        :param n_gen_stage2: generations in stage 2
        :param k_elites: number of elites for evolution in stage 1
        :param controlled_elitism: bool, whether to apply controlled elitism for NSGA-II, True by default
        :param method: 'mismatch' or 'distance', how to measure the difference between the model space and data space
        :param w_ssv: float, weight to penalize stable state violations. If none, then no penalty.
        :param parallel: bool, whether to parallelize the coevolution on multiple processes
        :param verbose: whether print information during evolution
        :return: list, representatives of each _species_list after evolution finishes
        """
        assert method in ['mismatch', 'distance'], "The method argument can only be 'mismatch' or 'distance'."
        self._verbose = verbose
        if parallel:
            self._coevolve_parallel(cxpb, mutpb, n_gen_stage1, n_gen_stage2, k_elites, controlled_elitism, method, w_ssv)
        else:
            self._coevolve_serial(cxpb, mutpb, n_gen_stage1, n_gen_stage2, k_elites,
                                  controlled_elitism, method, w_ssv)
        return self.logbook

    def _log_to_file(self, info):
        with open(self.log_file, 'w') as f:
            f.write(str(info))

    @property
    def species_list(self):
        return self._species_list
