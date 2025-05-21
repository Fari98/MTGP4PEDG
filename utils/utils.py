from collections import defaultdict
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import scipy.optimize
from scipy.optimize import rosen
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


def get_pareto_rankings(population):
    """
    Computes and assigns Pareto front ranks to individuals in the population.
    Minimizes both inutility and disclosure_averseness.

    Args:
        population: An object with a .individuals list. Each individual must have:
                    - .inutility
                    - .disclosure_averseness
                    - .front (will be set here)

    Returns:
        A list of integers representing the Pareto front (rank) for each individual
        in the original population order. Front 0 is the best (non-dominated).
    """
    individuals = population.individuals
    n = len(individuals)
    rankings = [-1] * n  # -1 means not assigned yet
    remaining = list(range(n))  # indices of individuals not yet ranked
    current_front = 0

    def dominates(a, b):
        return (
            a.inutility <= b.inutility and
            a.disclosure_averseness <= b.disclosure_averseness and
            (a.inutility < b.inutility or a.disclosure_averseness < b.disclosure_averseness)
        )

    while remaining:
        current_front_members = []
        for i in remaining:
            ind_i = individuals[i]
            is_dominated = False
            for j in remaining:
                if i == j:
                    continue
                ind_j = individuals[j]
                if dominates(ind_j, ind_i):
                    is_dominated = True
                    break
            if not is_dominated:
                current_front_members.append(i)

        for idx in current_front_members:
            individuals[idx].front = current_front
            rankings[idx] = current_front

        # Remove assigned individuals from remaining
        remaining = [i for i in remaining if i not in current_front_members]
        current_front += 1

    return rankings


def calculate_crowding_distances(population):
    """
    Assigns crowding distances to each Individual instance in the population.
    Each individual must have attributes:
    - inutility
    - disclosure_averseness
    - front (Pareto front index)

    The function adds/updates a `.crowding_distance` attribute for each individual.
    """

    # Group individuals by front
    fronts = defaultdict(list)
    for individual in population.individuals:
        fronts[individual.front].append(individual)

    for front_individuals in fronts.values():
        num_individuals = len(front_individuals)
        if num_individuals == 0:
            continue
        if num_individuals <= 2:
            for ind in front_individuals:
                ind.crowding_distance = float('inf')
            continue

        # Initialize distances
        for ind in front_individuals:
            ind.crowding_distance = 0.0

        objectives = ['inutility', 'disclosure_averseness']

        for obj in objectives:
            # Sort individuals by the objective
            front_individuals.sort(key=lambda ind: getattr(ind, obj))
            min_val = getattr(front_individuals[0], obj)
            max_val = getattr(front_individuals[-1], obj)

            # Assign infinite distance to boundary individuals
            front_individuals[0].crowding_distance = float('inf')
            front_individuals[-1].crowding_distance = float('inf')

            if max_val == min_val:
                continue  # avoid division by zero

            # Compute normalized distances for interior individuals
            for i in range(1, num_individuals - 1):
                prev_val = getattr(front_individuals[i - 1], obj)
                next_val = getattr(front_individuals[i + 1], obj)
                distance = (next_val - prev_val) / (max_val - min_val)
                if front_individuals[i].crowding_distance != float('inf'):
                    front_individuals[i].crowding_distance += distance

# @ignore_warnings(category=ConvergenceWarning)
def evaluate_dataset(real_space, real_res, synthetic_space, learning_techniques, clustering_technique, return_full_results = False):


    # real_results =  torch.from_numpy(np.array([cross_val_score(tech, real_space[:, :-1], real_space[:, -1], cv=5,
    #                                                               scoring='r2'
    #                                                               ) for tech in learning_techniques]))
    # real_res = torch.mean(real_results)

    # StratifiedKFold(n_splits=5, shuffle=True) #to change across runs, otherwise fixed
    synthetic_results = torch.from_numpy(np.array([cross_val_score(tech,
                                                                StandardScaler().fit_transform(synthetic_space[:, :-1]),
                                                                synthetic_space[:, -1],
                                                                scoring=bounded_r2_score,
                                                                cv = KFold(n_splits=5)
                                                                  ) for tech in learning_techniques]))

    syn_res = torch.mean(synthetic_results, dim = 1)

    inutility = torch.mean(torch.abs(torch.sub(syn_res, real_res))).item()

    complete_dataset = torch.concatenate(
        (torch.concatenate((synthetic_space, torch.zeros((synthetic_space.shape[0], 1))), dim=1),
         torch.concatenate((real_space, torch.ones((real_space.shape[0], 1))), dim=1)))

    unique_dataset, labels = torch.unique(complete_dataset[:, :-1], return_inverse=True, dim=0)
    unique_dataset = torch.concatenate((unique_dataset, complete_dataset[torch.unique(labels), -1].unsqueeze(1)), dim=1)


    clustering_technique.fit(unique_dataset[:, :-1])
    labels = clustering_technique.labels_
    unique_labels, counts = np.unique(clustering_technique.labels_, return_counts=True)

    label_counts = {}
    max_class_counts = []

    total_samples = unique_dataset.shape[0]

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        group_size = len(indices)

        label_counts[label] = group_size

        # Extract the target column (-1) for the current label
        target_values = unique_dataset[indices, -1]

        # Get counts of each unique value
        _, counts = np.unique(target_values, return_counts=True)
        max_count = counts.max()

        # Compute the term for this label
        max_class_counts.append((group_size / total_samples) * ((max_count / group_size) - 0.5))

    if return_full_results:

        return inutility, float(np.sum(max_class_counts)), synthetic_results.tolist()

    else:

        return inutility, float(np.sum(max_class_counts))

def non_zero_floor_division(a, b):

    if a < b:
        return 1
    else:
        return a // b

import torch

def sample_with_constant_handling(real_space):
    min_vals, _ = torch.min(real_space, dim=0)
    max_vals, _ = torch.max(real_space, dim=0)

    # Identify constant columns where min == max
    constant_mask = (min_vals == max_vals)

    # For non-constant columns only, create a Uniform distribution
    varying_min = min_vals[~constant_mask]
    varying_max = max_vals[~constant_mask]

    # Sample for non-constant columns
    uniform = torch.distributions.Uniform(varying_min, varying_max)
    sampled = uniform.sample([real_space.shape[0]])

    # Initialize random_space with same shape as real_space
    random_space = torch.empty_like(real_space)

    # Fill in sampled values for varying columns
    random_space[:, ~constant_mask] = sampled

    # Copy over constant columns from original data
    random_space[:, constant_mask] = real_space[:, constant_mask]

    return random_space

def smape(actual, predicted) -> float:

	# Convert actual and predicted to numpy
	# array data type if not already
	if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
		actual, predicted = np.array(actual), np.array(predicted)

	return  np.mean(
			np.abs(predicted - actual) /
			(((np.abs(predicted) + np.abs(actual))/2)) #+0.0000000001
		)

smape_score = make_scorer(smape)

def bounded_r2(actual, predicted):
    score = r2_score(actual, predicted)
    return score if score > -1 else -1.0

bounded_r2_score = make_scorer(bounded_r2)


def find_duplicate_indexes(data):
    seen = {}
    duplicates = []

    for idx, item in enumerate(data):
        if item in seen:
            duplicates.append(idx)
        else:
            seen[item] = idx

    return duplicates

def rastrigin(matrix):
    """
    Calculate the Rastrigin function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for xi in position:
            fitness_val += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
        fitness_values[i] = fitness_val
    return fitness_values

def ackley(matrix):
    """
    Calculate the Ackley function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        sum1 = 0.0
        sum2 = 0.0
        for xi in position:
            sum1 += xi * xi
            sum2 += math.cos(2 * math.pi * xi)
        avg1 = sum1 / n_cols
        avg2 = sum2 / n_cols
        fitness_val = -20 * math.exp(-0.2 * math.sqrt(avg1)) - math.exp(avg2) + 20 + math.e
        fitness_values[i] = fitness_val
    return fitness_values

def alpine_1(matrix):
    """
    Calculate the Alpine 1 function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for xi in position:
            fitness_val += abs(xi * math.sin(xi) + 0.1 * xi)
        fitness_values[i] = fitness_val
    return fitness_values

def alpine_2(matrix):
    """
    Calculate the Alpine 2 function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 1.0
        for xi in position:
            fitness_val *= math.sqrt(xi) * math.sin(xi)
        fitness_values[i] = fitness_val
    return fitness_values


def michalewicz(matrix, m=10):
    """
    Calculate the Michalewicz function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for idx, xi in enumerate(position):
            fitness_val -= math.sin(xi) * (math.sin((idx + 1) * xi ** 2 / math.pi)) ** (2 * m)
        fitness_values[i] = fitness_val
    return fitness_values


def sphere(matrix):
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for xi in position:
            fitness_val += (xi * xi)
        fitness_values[i] = fitness_val
    return fitness_values

def create_random_matrix(n, m, ranges):
    if len(ranges) != m:
        raise ValueError("Length of ranges list must be equal to the number of columns (m).")
    matrix = np.zeros((n, m))
    for i in range(m):
        min_val, max_val = ranges[i]
        matrix[:, i] = np.random.uniform(min_val, max_val, n)
    return matrix


def create_dataset(rows, columns, function, seed):

    np.random.seed(seed)

    functions = {'rastrigin' : rastrigin,
                 'sphere' : sphere,
                 'rosenbrock' : scipy.optimize.rosen,
                 'ackley' : ackley,
                 'alpine1' : alpine_1,
                 'alpine2' : alpine_2,
                 'michalewicz' : michalewicz
                 }

    if function not in functions.keys():
        raise ValueError('Invalid function')

    if function == 'rastrigin':
        X = create_random_matrix(rows, columns, [(-5.12, 5.12) for _ in range(columns)])

    elif function == 'alpine2':
        X = create_random_matrix(rows, columns, [(0, 10) for _ in range(columns)])
    else:
        X = create_random_matrix(rows, columns, [(-10,10) for _ in range(columns)])

    if function == 'rosenbrock':
        y = functions[function](X.T).reshape(-1, 1)
    else:
        y = functions[function](X).reshape(-1, 1)

    return X, y.flatten()