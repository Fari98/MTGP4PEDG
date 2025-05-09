from tree.mutation_operators import mutate_tree_subtree
from tree.tree import Tree
from individual.individual import Individual
import random

def element_wise_mutation(parent):

    mutation = mutate_tree_subtree(3, Tree.TERMINALS, Tree.CONSTANTS, Tree.FUNCTIONS, p_c = 0.1)

    return Individual([mutation(parent.representations[i], parent.trees[i].nodes  ) for i in range(parent.lenght)])

def uniform_mutation(parent):
    mask = [random.choice([0, 1]) for _ in range(parent.lenght)]
    print(mask)
    
    mutation = mutate_tree_subtree(3, Tree.TERMINALS, Tree.CONSTANTS, Tree.FUNCTIONS, p_c=0.1)
    
    return Individual([mutation(parent.representations[i], parent.trees[i].nodes  ) if mask[i] == 1 else parent.representations[i] for i in range(parent.lenght)])


