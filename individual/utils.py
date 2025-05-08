from utils.utils import evaluate_dataset

def _evaluate_individual(individual, real_space, real_res, latent_space, learning_techniques, clustering_technique, return_full_dataset = False):
    
    synthetic_dataset = individual.predict(latent_space)

    return evaluate_dataset(real_space, real_res, synthetic_dataset, learning_techniques, clustering_technique, return_full_results= return_full_dataset)