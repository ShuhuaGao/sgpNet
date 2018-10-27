"""
Utilities for genetic programming specially designed for this Boolean network coevolution task.
"""
import operator
from deap import gp
import sympy as sp
import graphviz as gv
import copy


class NetPrimitiveSet(gp.PrimitiveSet):
    """
    A specialized primitive set used in Boolean network inference with coevolution.
    """
    def __init__(self, target_index, candidate_indices, genes):
        """
        Build a primitive set instance for the target target_gene.

        :param target_index: species_index of the target target_gene
        :param candidate_indices: indices the possible genes as regulators of target target_gene
        """
        self.target_index = target_index
        self._candidate_indices = list(candidate_indices)  # candidate genes
        self.genes = genes
        super().__init__(self.target_gene, len(candidate_indices))
        self._init_primitive_set()

    def _init_primitive_set(self):
        # function set: AND, OR, NOT
        self.addPrimitive(operator.and_, 2)
        self.addPrimitive(operator.or_, 2)
        self.addPrimitive(operator.not_, 1)
        # terminal set: only inputs. Rename the arguments to be the candidate names
        self.renameArguments(**{f'ARG{i}': candidate for i, candidate in enumerate(self.candidate_genes)})

    @property
    def target_gene(self):
        """
        Get the target target_gene.

        :return: target target_gene
        """
        return self.genes[self.target_index]

    @property
    def candidate_genes(self):
        """
        Get the candidate regulator genes.

        :return: a list of genes
        """
        return [self.genes[i] for i in self.candidate_indices]

    @property
    def candidate_indices(self):
        """
        Get the indices for the possible regulators.

        :return:
        """
        return self._candidate_indices


def simplify(ind: gp.PrimitiveTree, pset: gp.PrimitiveSet, symbol_map=None):
    """
    Compile the primitive tree into a (possibly simplified) symbolic expression

    :param ind: a primitive tree
    :param pset: a primitive set
    :param symbol_map: map each function name in the primitive set to a symbolic version.
        If ``None``, use a default one.
    :return: a (simplified) symbol expression corresponding to the given PrimitiveTree
    """
    assert isinstance(ind, gp.PrimitiveTree)
    from sympy.logic import simplify_logic
    if symbol_map is None:
        symbol_map = {operator.and_.__name__: sp.And,
                      operator.or_.__name__: sp.Or,
                      operator.not_.__name__: sp.Not}
    operand_stack = []
    r = None
    # the elements in gp.PrimitiveTree in fact represents a prefix expression
    for node in reversed(ind):
        if isinstance(node, gp.Ephemeral):  # a randomly generated constant
            operand_stack.append(node.value)
        elif isinstance(node, gp.Terminal):
            # whether this terminal represents an input?
            if node.value in pset.arguments:
                operand_stack.append(sp.Symbol(node.value))
            else:  # just a constant
                operand_stack.append(node.value)
        elif isinstance(node, gp.Primitive):  # function
            sym_func = symbol_map[node.name]  # get the symbolic version of this function
            try:
                args = [operand_stack.pop() for _ in range(node.arity)]
                r = sym_func(*args)
                r = simplify_logic(r)
                operand_stack.append(r)
            except AttributeError as err:
                print(err)
                print(sym_func)
                print(args)
                print(type(arg) for arg in args)
        else:
            raise RuntimeError('Not recognized node type in the primitive tree: {}'.format(type(node)))
    return operand_stack.pop()


def export_tree(tree, output_file_without_extension, extension='png', view=False):
    """
    Visualize a gp tree by exporting it into an image.
    :param tree: gp.PrimitiveTree
    :param output_file_without_extension: file path, for example, './img/tree'
    :param extension: specify image file type, for example, '.bmp'
    :param view: whether to show the image automatically
    """
    # NOTE:
    # nodes are integers indexed from 0
    # edges [(0, 1), (3, 2), ...]
    # labels is a dict: {0: 'A', 1: 'ADD', 2: -1...}. Note values may be numbers (constants).
    # in graphviz package, only string name/label are allowed
    nodes, edges, labels = gp.graph(tree)
    g = gv.Graph(format=extension)
    for name, label in labels.items():
        g.node(str(name), str(label))  # add node
    for name1, name2 in edges:
        g.edge(str(name1), str(name2))  # add edge
    g.render(output_file_without_extension, view=view)


class Archive:
    """
    Store and update the non-dominated solutions found so far in multi-objective evolution.

    ..note:
      `deap.HallOfFame` simply uses lexicographical comparison even for multi-objective problems.
    """
    def __init__(self, maxsize):
        self._maxsize = maxsize
        self._solutions = []

    @property
    def maxsize(self):
        return self._maxsize

    def update(self, population):
        """
        Update the archive with the *population* according to fitness dominance. The size of the archive is kept
        constant specified by :attr:`maxsize`.

        :param population: a list of individuals with a *fitness* attribute
        """
        for ind in population:
            # 1. whether ind is dominated by any solution
            is_dominated = False
            for sol in self._solutions:
                if sol.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
            # 2. remove the solutions dominated by ind
            if not is_dominated:
                to_delete = []
                for i, sol in enumerate(self._solutions):
                    if ind.fitness.dominates(sol.fitness):
                        to_delete.append(i)
                for i in reversed(to_delete):
                    del self._solutions[i]
            # 3. append ind if eligible
            if not is_dominated:
                self._solutions.append(copy.deepcopy(ind))
            # 4. remove one solution if the max size is exceeded
            if len(self._solutions) > self.maxsize:
                self._remove()

    def _remove(self):
        """
        Remove one individual/solution from the archive to maintain the constant size.
        Since all individuals are nondominated in this archive, we remove the minimum one by lexicographical ordering.
        That is, remove the one with the worst global fitness.
        """
        index = min(range(len(self._solutions)), key=lambda i: self._solutions[i].fitness)
        del self._solutions[index]

    def __getitem__(self, item):
        return self._solutions[item]

    def __iter__(self):
        return iter(self._solutions)

    def __reversed__(self):
        return reversed(self._solutions)

    def __len__(self):
        return len(self._solutions)

    def clear(self):
        self._solutions.clear()

    def sort(self, which_obj=0, reverse=True):
        """
        Sort the solutions in this archive according to the specified objective.
        :param which_obj: int, the objective index. Default: 0.
        :param reverse: True: best to worst; False: worst to best. Default: True.
        """
        self._solutions.sort(key=lambda sol: sol.fitness.values[which_obj] * sol.fitness.weights[which_obj],
                             reverse=reverse)





