from utils.utils import get_pareto_rankings, calculate_crowding_distances
import random
import numpy as np

def nsga_II(population):

    if population.fronts is None:
        population.fronts = get_pareto_rankings(population)

    if population.crowding_distances is None:
        calculate_crowding_distances(population)
        population.crowding_distances = [ind.crowding_distance for ind in population.individuals]

    pool = random.choices(population.individuals, k = 2)

    if pool[0].front != pool[1].front:
        return pool[np.argmin([ind.front for ind in pool])]
    else:
        return pool[np.argmax([ind.crowding_distance for ind in pool])]





