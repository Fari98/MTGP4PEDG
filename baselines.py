from utils.utils import evaluate_dataset, sample_with_constant_handling
from datasets.data_loader import load_concrete_strength
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import HDBSCAN
import torch
import numpy as np
import random
from datasets.data_loader import load_concrete_strength, load_bioav, load_airfoil, load_boston
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

results = {}
for loader in [
    load_concrete_strength,
               load_boston,
    load_airfoil,
    load_bioav
]:
    print(loader.__name__.split("load_")[-1])

    real_space = torch.from_numpy(loader(X_y = False).values)
    # random_space = torch.randn(real_space.shape)
    # random_space = real_space[:,:-1] + torch.randn(real_space[:,:-1].shape)
    # random_space = torch.concatenate((random_space, real_space[:,-1].unsqueeze(1)), dim = 1)

    for tech in [RandomForestRegressor(), SVR(), LinearRegression(), DecisionTreeRegressor(), XGBRegressor()]:

        print(str(tech))

        cv = cross_val_score(tech, real_space[:,:-1], real_space[:,-1], cv = 5)

        print(cv)
        # print(cv['score_time'])

    # random_space = real_space[:,:-1] + torch.var(real_space[:,:-1], dim = 0)*torch.randn(real_space[:,:-1].shape)
    # random_space = torch.concatenate((random_space, torch.randn(real_space[:,-1].shape).unsqueeze(1)), dim = 1)

    # random_space = sample_with_constant_handling(real_space)
    #
    #
    # max_utility = evaluate_dataset(real_space=real_space,
    #                         synthetic_space=real_space,
    #                         learning_techniques=[RandomForestRegressor(), SVR()],
    #                         clustering_technique=HDBSCAN())
    #
    # max_da = evaluate_dataset(real_space=real_space,
    #                         synthetic_space=random_space,
    #                         learning_techniques=[RandomForestRegressor(), SVR()],
    #                         clustering_technique=HDBSCAN())
    #
    # results[loader.__name__.split("load_")[-1]] = [max_utility, max_da]

print(results)