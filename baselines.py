from utils.utils import evaluate_baseline
from datasets.data_loader import load_concrete_strength
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import HDBSCAN
import torch

real_space = torch.from_numpy(load_concrete_strength(X_y = False).values)
# random_space = torch.randn(real_space.shape)
# random_space = real_space[:,:-1] + torch.randn(real_space[:,:-1].shape)
# random_space = torch.concatenate((random_space, real_space[:,-1].unsqueeze(1)), dim = 1)

# random_space = real_space[:,:-1] + torch.var(real_space[:,:-1], dim = 0)*torch.randn(real_space[:,:-1].shape)
# random_space = torch.concatenate((random_space, torch.randn(real_space[:,-1].shape).unsqueeze(1)), dim = 1)

random_space = torch.distributions.Uniform(torch.min(real_space, dim = 0)[0], torch.max(real_space, dim =0)[0]).sample([real_space.shape[0]])


print(evaluate_baseline(real_space=real_space,
                        synthetic_space=real_space,
                        learning_techniques=[RandomForestRegressor(), SVR()],
                        clustering_technique=HDBSCAN()))

print(evaluate_baseline(real_space=real_space,
                        synthetic_space=random_space,
                        learning_techniques=[RandomForestRegressor(), SVR()],
                        clustering_technique=HDBSCAN()))