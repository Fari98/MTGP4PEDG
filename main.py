from MTGP4SDG import MTGP4SDG
from population.initializers import initialize_multitree_population
from tree.initializers import rhh
from individual.crossover_operators import uniform_crossover
from individual.mutation_operators import element_wise_mutation
from utils.functions import FUNCTIONS
from utils.constants import CONSTANTS
from utils.terminals import get_terminals
from population.selection_algoritmhs import nsga_II
from datasets.data_loader import load_concrete_strength, load_bioav, load_airfoil, load_boston
import torch
import random
import numpy as np
from tree.tree import Tree

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import HDBSCAN
from utils.utils import non_zero_floor_division

# loader = load_concrete_strength

for loader in [
    # load_concrete_strength,
    #            load_boston,
    load_airfoil,
    # load_bioav
]:

    real_space = torch.from_numpy(loader(X_y=False).values)
    dataset = loader.__name__.split("load_")[-1]

    uniform_half = torch.rand((real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8))) * 2 - 1  # Scaled to [-1, 1]
    # Half from Gaussian N(0, 1)
    gaussian_half = torch.randn((real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8)))
    # Concatenate along columns
    latent_space = torch.cat([uniform_half, gaussian_half], dim=1)

    # latent_space = torch.distributions.multivariate_normal.MultivariateNormal(
    #     loc=torch.zeros(int(real_space.shape[1]/4)),
    #     covariance_matrix=torch.eye(int(real_space.shape[1]/4))).sample([real_space.shape[0]])

    for seed in range(5):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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
                pop_size=20,
                seed=seed)


        generator.solve(real_space,
                latent_space,
                [RandomForestRegressor(), SVR()],
                HDBSCAN(),
                generations=50,
                elitism=True,
                dataset_name=dataset,
                log=1,
                log_path = 'log/first_experiment.csv',
                verbose=1,
                n_jobs = -1)