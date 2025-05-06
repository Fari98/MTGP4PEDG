import random
import time
import numpy as np
import torch

from utils.info import logger, verbose_reporter

from population.population import Population



class MTGP4SDG:
    def __init__(
        self,
        initializer,
        selector,
        mutator,
        crossover,
        p_m=0.2,
        p_xo=0.8,
        pop_size=100,
        seed=0,
    ):
        """
        Initialize the Genetic Programming algorithm.

        Parameters
        ----------
ì
        """
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed

    def solve(
        self,
        real_space,
        latent_space,
        learning_techniques,
        clustering_technique,
        generations=20,
        elitism=True,
        dataset_name=None,
        log=0,
        log_path = None,
        verbose=0,
        n_jobs = 1
    ):
        """
        Execute the Genetic Programming algorithm.

        Parameters
        ----------

        """
        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # Initialize the population
        self.population = Population(self.initializer(self.pop_size))

        # evaluating the intial population
        self.population.evaluate(real_space, latent_space,
                            learning_techniques, clustering_technique, n_jobs)

        end = time.time()

        self.elites = self.population.find_elites()

        # logging the results if the log level is not 0

        timing = end-start

        if log != 0:
            logger(log_path,
                   0,
                   timing,
                   [[individual.utility for individual in self.elites],
                    [individual.disclosure_averseness for individual in self.elites],
                    # [individual.representations for individual in self.elites]
                    ],
                   self.seed)

        # displaying the results on console if verbose level is not 0
        if verbose != 0:
            verbose_reporter(
                dataset_name,
                0,
                min([individual.utility for individual in self.elites]),
                min([individual.disclosure_averseness for individual in self.elites]),
                timing
            )

        # EVOLUTIONARY PROCESS
        for generation in range(1, generations + 1):

            start = time.time()

            if elitism:
                offs_pop = self.elites
            else:
                offs_pop = []
            while len(offs_pop) < self.population.size:

                if random.random() < self.p_m:

                    parent = self.selector(self.population)
                    offspring = self.mutator(parent)

                    offs_pop.append(offspring)

                else:

                    p1, p2 = self.selector(self.population), self.selector(self.population)
                    offs1, offs2 = self.crossover(p1, p2)

                    offs_pop.extend([offs1, offs2])

            offs_pop = offs_pop[:self.population.size]
            offs_pop = Population(offs_pop)
            # replacing the population with the offspring population (P = P')
            self.population = offs_pop

            if elitism:
                [individual.__setattr__('front', None) for individual in self.population.individuals]

            self.population.evaluate(real_space, latent_space,
                            learning_techniques, clustering_technique, n_jobs)

            # getting the new elite(s)
            self.elites = self.population.find_elites()

            end = time.time()

            timing = end - start

            if log != 0:
                logger(log_path,
                       generation,
                       timing,
                       [[individual.utility for individual in self.elites],
                        [individual.disclosure_averseness for individual in self.elites],
                        # [individual.representations for individual in self.elites]
                        ],
                       self.seed)

            # displaying the results on console if verbose level is not 0
            if verbose != 0:
                verbose_reporter(
                    dataset_name,
                    generation,
                    min([individual.utility for individual in self.elites]),
                    min([individual.disclosure_averseness for individual in self.elites]),
                    timing
                )