from joblib import Parallel, delayed
from individual.utils import _evaluate_individual
from utils.utils import get_pareto_rankings


class Population:
    def __init__(self, individuals):

        self.individuals = individuals
        self.size = len(individuals)

        self.fronts = None
        self.crowding_distances = None

    def evaluate(self, real_space, latent_space,
                learning_techniques, clustering_technique, n_jobs=1):
        """
        Evaluates the population given a certain fitness function, input data (X), and target data (y).

        Attributes a fitness tensor to the population.

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individuals.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        n_jobs : int
            The maximum number of concurrently running jobs for joblib parallelization.

        Returns
        -------
        None
        """
        # Evaluates individuals' semantics
        fitness_values = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_individual)(
                individual, real_space, latent_space,
                learning_techniques, clustering_technique
            ) for individual in self.individuals
        )

        self.utilities = [fitness[0] for fitness in fitness_values]
        self.disclosure_aversenesses = [fitness[1] for fitness in fitness_values]

        self.fitness_values = fitness_values

        # Assign individuals' fitness
        [self.individuals[i].__setattr__('utility', u) for i, u in enumerate(self.utilities)]
        [self.individuals[i].__setattr__('disclosure_averseness', da) for i, da in enumerate(self.disclosure_aversenesses)]



    def find_elites(self):

        if self.fronts is None:
            self.fronts = get_pareto_rankings(self)

        return [self.individuals[i] for i in range(self.size) if self.fronts[i] == 0]
