from utils.utils import evaluate_dataset, sample_with_constant_handling, smape
from sklearn.metrics import make_scorer
from datasets.data_loader import load_concrete_strength
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import HDBSCAN
import torch
import numpy as np
import random
from datasets.data_loader import load_concrete_strength, load_bioav, load_airfoil, load_boston
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from utils.utils import smape_score, bounded_r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.info import base_logger
import datetime

now = datetime.datetime.now()
day = now.strftime("%Y%m%d")


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
#
# for loader in [
#     # load_concrete_strength,
#     # load_boston,
#     load_airfoil,
#     # load_bioav
# ]:

loaders = [
    load_concrete_strength,
    load_boston,
    load_airfoil,
    load_bioav
]

def _run(syn_space_path, loader):

    dataset = loader.__name__.split("load_")[-1]

    real_space = torch.from_numpy(loader(X_y = False).values)

    synthetic_space = pd.read_csv(syn_space_path).values


    train_techniques = [MLPRegressor(max_iter = 2000, random_state=0),
              KNeighborsRegressor(),
              RandomForestRegressor(random_state=0),
              # XGBRegressor(device = 'cpu', random_state=0)
              ]

    real_results = []
    for tech in train_techniques:
        real_results.append(cross_val_score(tech, StandardScaler().fit_transform(real_space[:,:-1]), real_space[:, -1],
                                        scoring = bounded_r2_score,
                                        cv = KFold(n_splits=5)
                                            ))

    real_res = torch.mean(torch.from_numpy(np.array(real_results)), dim = 1)

    syn_results = []
    for tech in train_techniques:
        syn_results.append(cross_val_score(tech, StandardScaler().fit_transform(synthetic_space[:, :-1]), synthetic_space[:, -1],
                                        scoring = bounded_r2_score,
                                        cv = KFold(n_splits=5)
                                           ))

    syn_res = torch.mean(torch.from_numpy(np.array(syn_results)), dim = 1)


    train_perf = torch.mean(torch.abs(torch.sub(real_res, syn_res))).item()

    val_techniques = [
                  DecisionTreeRegressor(),
                  LinearRegression(),
                  XGBRegressor(device='cpu', random_state=0),
                  SVR(),
                  Lasso(),
                  Ridge(),
                  ExtraTreeRegressor(),
                  ElasticNet(),
                  HistGradientBoostingRegressor()
                 ]

    real_results = []
    for tech in val_techniques:
        real_results.append(cross_val_score(tech, StandardScaler().fit_transform(real_space[:,:-1]), real_space[:, -1],
                                        scoring = bounded_r2_score,
                                        cv = KFold(n_splits=5)
                                            ))

    real_res = torch.mean(torch.from_numpy(np.array(real_results)), dim = 1)

    syn_results = []
    for tech in val_techniques:
        syn_results.append(cross_val_score(tech, StandardScaler().fit_transform(synthetic_space[:, :-1]), synthetic_space[:, -1],
                                        scoring = bounded_r2_score,
                                        cv = KFold(n_splits=5)
                                           ))

    syn_res = torch.mean(torch.from_numpy(np.array(syn_results)), dim = 1)


    val_perf = torch.mean(torch.abs(torch.sub(real_res, syn_res))).item()


    base_logger(
           path=f'log/compare_nlt_{day}.csv',
           row=[dataset, train_perf, val_perf, syn_space_path.split('_')[-2], syn_space_path.split('_')[-1]]
           )



