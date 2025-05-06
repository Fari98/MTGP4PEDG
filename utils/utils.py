from collections import defaultdict
import numpy as np
from sklearn.model_selection import cross_validate
import torch

def get_pareto_rankings(population):
    """
    Computes and assigns Pareto front ranks to individuals in the population.
    Minimizes both utility and disclosure_averseness.

    Args:
        population: An object with a .individuals list. Each individual must have:
                    - .utility
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
            a.utility <= b.utility and
            a.disclosure_averseness <= b.disclosure_averseness and
            (a.utility < b.utility or a.disclosure_averseness < b.disclosure_averseness)
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
    - utility
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

        objectives = ['utility', 'disclosure_averseness']

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

def evaluate_dataset(real_space, synthetic_space, learning_techniques, clustering_technique):


    real_res = torch.median(torch.from_numpy(np.array([cross_validate(tech, real_space[:, :-1], real_space[:, -1], cv=5,
                                                                  scoring='r2'
                                                                  )['test_score'] for tech in learning_techniques])))
    syn_res = torch.median(torch.from_numpy(np.array([cross_validate(tech, synthetic_space[:, :-1], synthetic_space[:, -1], cv=5,
                                                                  scoring='r2'
                                                                  )['test_score'] for tech in learning_techniques])))

    utility = torch.abs(torch.sub(syn_res, real_res)).item()

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

    return utility, float(np.sum(max_class_counts))


