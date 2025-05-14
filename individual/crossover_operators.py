import random
from tree.crossover_operators import crossover_trees
from tree.tree import Tree
from individual.individual import Individual

def uniform_crossover(p1, p2, max_allowed_depth = None):

    mask = [random.choice([0,1]) for _ in range(p1.lenght)]

    return (Individual([p1.representations[i] if mask[i] == 1 else p2.representations[i] for i in range(len(mask))]),
            Individual([p2.representations[i] if mask[i] == 1 else p1.representations[i] for i in range(len(mask))]))

def element_wise_crossover(p1, p2, max_allowed_depth = None):

    xo = crossover_trees(Tree.FUNCTIONS, max_allowed_depth = max_allowed_depth)

    offs = [(xo(p1.representations[i], p2.representations[i],
               p1.trees[i].nodes, p2.trees[i].nodes)) for i in range(p1.lenght)]

    off1, off2 = zip(*offs)

    return Individual(off1), Individual(off2)