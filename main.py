from MTGP4SDG import MTGP4SDG
from population.initializers import initialize_multitree_population
from tree.initializers import rhh
from individual.crossover_operators import uniform_crossover
from individual.mutation_operators import element_wise_mutation
from utils.functions import FUNCTIONS
from utils.constants import CONSTANTS
from utils.terminals import get_terminals
from population.selection_algoritmhs import nsga_II
from datasets.data_loader import load_concrete_strength
import torch
import random
import numpy as np
from tree.tree import Tree

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import HDBSCAN

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

real_space = torch.from_numpy(load_concrete_strength(X_y = False).values)
latent_space = torch.distributions.multivariate_normal.MultivariateNormal(
                                                    loc = torch.zeros(2),
                                                    covariance_matrix=torch.eye(2)).sample([real_space.shape[0]])

TERMINALS = get_terminals(latent_space)
Tree.FUNCTIONS = FUNCTIONS
Tree.TERMINALS = TERMINALS
Tree.CONSTANTS = CONSTANTS

generator = MTGP4SDG(initialize_multitree_population(individual_size = real_space.shape[1],
                                    initial_depth = 6,
                                    technique = rhh,
                                    FUNCTIONS = FUNCTIONS,
                                    TERMINALS = TERMINALS,
                                    CONSTANTS = CONSTANTS),
        selector = nsga_II,
        mutator = element_wise_mutation,
        crossover = uniform_crossover,
        p_m=0.2,
        p_xo=0.8,
        pop_size=10,
        seed=0)


generator.solve(real_space,
        latent_space,
        [RandomForestRegressor(), SVR()],
        HDBSCAN(),
        generations=20,
        elitism=True,
        dataset_name='concrete_strength',
        log=0,
        log_path = None,
        verbose=1,
        n_jobs = -1)