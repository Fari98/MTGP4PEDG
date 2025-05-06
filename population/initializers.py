import random
from tree.tree import Tree
from individual.individual import Individual

def initialize_multitree_population(individual_size:int,
                                    initial_depth:int,
                                    technique,
                                    FUNCTIONS:dict,
                                    TERMINALS:dict,
                                    CONSTANTS: dict) -> None:

    def inner_initializer(population_size):

        total_individuals = int(individual_size * population_size)
        pool = technique(total_individuals, initial_depth, FUNCTIONS, TERMINALS, CONSTANTS)
        # pool = [Tree(individual) for individual in pool] #done later in the INdividual creation process
        #shuffling list
        pool = random.sample(pool, len(pool))

        #dividing it in sublists
        return [Individual(pool[i*individual_size : (i+1)*individual_size]) for i in range(population_size)]

    return inner_initializer

