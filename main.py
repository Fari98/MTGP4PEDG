from MTGP4EDG import MTGP4PEDG
from population.initializers import initialize_multitree_population
from tree.initializers import rhh
from individual.crossover_operators import uniform_crossover
from individual.mutation_operators import element_wise_mutation
from utils.functions import FUNCTIONS
from utils.constants import CONSTANTS
from utils.terminals import get_terminals
from population.selection_algoritmhs import nsga_II
from datasets.data_loader import load_boston
import torch
import random
import numpy as np
from tree.tree import Tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from utils.utils import bounded_r2_score, non_zero_floor_division


techniques = [MLPRegressor(max_iter = 2000, random_state=0),
              KNeighborsRegressor(),
              RandomForestRegressor(random_state=0),
              ]

real_space = torch.from_numpy(load_boston(X_y=False).values)
dataset = 'boston'

real_res = []
for tech in techniques:
    real_res.append(cross_val_score(tech, StandardScaler().fit_transform(real_space[:,:-1]), real_space[:, -1],
                                    scoring = bounded_r2_score, cv = 5))

real_res = torch.mean(torch.from_numpy(np.array(real_res)))

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
 

uniform_half = torch.rand((real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8))) * 2 - 1
gaussian_half = torch.randn((real_space.shape[0],  non_zero_floor_division(real_space.shape[1], 8)))
latent_space = torch.cat([uniform_half, gaussian_half], dim=1)

TERMINALS = get_terminals(latent_space)
Tree.FUNCTIONS = FUNCTIONS
Tree.TERMINALS = TERMINALS
Tree.CONSTANTS = CONSTANTS

generator = MTGP4PEDG(initialize_multitree_population(individual_size = real_space.shape[1],
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
        pop_size=50,
        seed=0)


generator.solve(real_space,
        real_res,
        latent_space,
        techniques,
        HDBSCAN(),
        max_depth= 17,
        generations=50,
        elitism=True,
        remove_copies= True,
        dataset_name=dataset,
        log=1,
        log_path = f'log/experiment.csv',
        verbose=1,
        n_jobs = 1)

