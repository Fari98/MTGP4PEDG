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
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.cluster import HDBSCAN
from utils.utils import non_zero_floor_division, smape_score, bounded_r2_score, create_dataset
import os
import csv
from sklearn.decomposition import PCA

import datetime



now = datetime.datetime.now()
day = now.strftime("%Y%m%d")

techniques = [MLPRegressor(max_iter = 2000, random_state=0),
              KNeighborsRegressor(),
              RandomForestRegressor(random_state=0),
              # XGBRegressor(device = 'cpu', random_state=0)
              ]

loaders = [
    # load_concrete_strength,
    load_boston,
    # load_airfoil,
    # load_bioav,
    # create_dataset(1000, 1, 'rastrigin', 0)
]
#
seeds = 5

# real_space = torch.from_numpy(loaders[0](X_y=False).values)
# dataset = loaders[0].__name__.split("load_")[-1]

def _run(seed, loader):

    real_space = torch.from_numpy(loader(X_y=False).values)
    dataset = loader.__name__.split("load_")[-1]

    # latent_space = torch.distributions.multivariate_normal.MultivariateNormal(
    #     loc=torch.zeros(int(real_space.shape[1]/4)),
    #     covariance_matrix=torch.eye(int(real_space.shape[1]/4))).sample([real_space.shape[0]])

    real_res = []
    for tech in techniques:
        real_res.append(cross_val_score(tech, StandardScaler().fit_transform(real_space[:,:-1]), real_space[:, -1],
                                        scoring = bounded_r2_score, cv = 5))

    real_res = torch.mean(torch.from_numpy(np.array(real_res)))



    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # for creation in ['other_distributions', 'only_gaussian']: #, 'other_distributions', 'std',, 'only_gaussian', 'pca', 'pca_automation',  'same_shape'
    #
    #     if creation == 'only_uniform':
    #         latent_space = torch.rand((real_space.shape[0], 1)) * 2 - 1
    #     elif creation == 'only_gaussian':
    #         latent_space = torch.randn((real_space.shape[0], 1))
    #     elif creation == 'pca':
    #         latent_space = torch.from_numpy(PCA(n_components=0.9).fit_transform(real_space))
    #     elif creation == 'pca_automation':
    #         n_features = PCA(n_components=0.9).fit_transform(real_space).shape[1]
    #         uniform_half = torch.rand(
    #             (real_space.shape[0], non_zero_floor_division(n_features, 2))) * 2 - 1  # Scaled to [-1, 1]
    #         # Half from Gaussian N(0, 1)
    #         gaussian_half = torch.randn((real_space.shape[0], non_zero_floor_division(n_features, 2)))
    #         # Concatenate along columns
    #         latent_space = torch.cat([uniform_half, gaussian_half], dim=1)
    #     elif creation == 'std':
    #         uniform_half = torch.rand(
    #             (real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8))) * 2 - 1  # Scaled to [-1, 1]
    #         # Half from Gaussian N(0, 1)
    #         gaussian_half = torch.randn((real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8)))
    #         # Concatenate along columns
    #         latent_space = torch.cat([uniform_half, gaussian_half], dim=1)
    #     elif creation == 'same_shape':
    #         uniform_half = torch.rand(
    #             (real_space.shape[0], non_zero_floor_division(real_space.shape[1], 2))) * 2 - 1  # Scaled to [-1, 1]
    #         # Half from Gaussian N(0, 1)
    #         gaussian_half = torch.randn((real_space.shape[0], non_zero_floor_division(real_space.shape[1], 2)))
    #         # Concatenate along columns
    #         latent_space = torch.cat([uniform_half, gaussian_half], dim=1)
    #     elif creation == 'other_distributions':
    #         exp_raw = torch.distributions.Exponential(rate=1.0).sample(
    #             (real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8)))
    #         exp_flat = exp_raw.flatten()
    #
    #         # Min-max scaling to [0, 1]
    #         exp_min = exp_raw.min()
    #         exp_max = exp_raw.max()
    #         exp_scaled = (exp_raw - exp_min) / (exp_max - exp_min)
    #
    #         # Bimodal Gaussian distribution: half N(-0.5, 0.1), half N(0.5, 0.1)
    #         bimodal_1 = torch.randn((real_space.shape[0]//2, non_zero_floor_division(real_space.shape[1], 8))) * 0.1 - 0.5
    #         bimodal_2 = torch.randn((real_space.shape[0]//2, non_zero_floor_division(real_space.shape[1], 8))) * 0.1 + 0.5
    #         bimodal = torch.cat([bimodal_1, bimodal_2], dim=0)
    #
    #         # Final latent space
    #         latent_space = torch.cat([exp_scaled, bimodal], dim=1)
    #     else:
    #         raise Exception('invalid latent space creation method')


    for size in [8, 28, 56, 140]:

        # uniform_half = torch.rand(
        #     (real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8))) * 2 - 1  # Scaled to [-1, 1]
        # # Half from Gaussian N(0, 1)
        # gaussian_half = torch.randn((real_space.shape[0], non_zero_floor_division(real_space.shape[1], 8)))
        # # Concatenate along columns
        # latent_space = torch.cat([uniform_half, gaussian_half], dim=1)

        uniform_half = torch.rand(
            (real_space.shape[0], size//2)) * 2 - 1  # Scaled to [-1, 1]
        # Half from Gaussian N(0, 1)
        gaussian_half = torch.randn((real_space.shape[0],  size//2))
        # Concatenate along columns
        latent_space = torch.cat([uniform_half, gaussian_half], dim=1)

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
                pop_size=50,
                seed=seed)


        generator.solve(real_space,
                real_res,
                latent_space,
                techniques,
                HDBSCAN(),
                max_depth= 17,
                generations=50,
                elitism=True,
                remove_copies= True,
                dataset_name=dataset + '_' + str(size),
                log=1,
                log_path = f'log/ls_size_evolution_{day}.csv',
                verbose=1,
                n_jobs = -1)

        # if generator.log > 0:
        #
        #     path = f'log/final_{day}.csv'
        #     if not os.path.isdir(os.path.dirname(path)):
        #         os.mkdir(os.path.dirname(path))
        #     with open(path, "a", newline="") as file:
        #         writer = csv.writer(file)
        #         writer.writerow([seed, dataset] + [individual.representations for individual in generator.elites])
        #
        #     if not os.path.isdir(f'log/{day}'):
        #         os.mkdir(f'log/{day}')
        #     [pd.DataFrame(individual.predict(latent_space)).to_csv(f'log/{day}/{dataset}_{seed}_{i}.csv') for i, individual in enumerate(generator.elites)]

