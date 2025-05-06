from utils.utils import evaluate_dataset

def _evaluate_individual(individual, real_space, latent_space, learning_techniques, clustering_technique):
    
    synthetic_dataset = individual.predict(latent_space)

    return evaluate_dataset(real_space, synthetic_dataset, learning_techniques, clustering_technique)