from tree.tree import Tree
import torch
from individual.utils import _evaluate_individual


class Individual:

    def __init__(self, representations, inutility = None, disclosure_averseness = None,
                                        # utility_test = None, disclosure_averseness_test = None #todo check wheter to include
                 ):


        self.trees = [Tree(tree) if not isinstance(tree, Tree) else tree for tree in representations]
        self.representations = [tree.repr_ if isinstance(tree, Tree) else tree for tree in representations]

        self.nodes = sum([tree.nodes for tree in self.trees])

        self.lenght = len(representations)

        self.inutility = inutility
        self.discolsure_averseness = disclosure_averseness

        # todo nodes depth information ??


    def predict(self, latent_space):

        return torch.concatenate([tree.predict(latent_space).unsqueeze(1) for tree in self.trees], dim=1)


    def evaluate(self, real_space, latent_space, learning_techniques, clustering_technique, full_results = False):

        if self.inutility is None and self.discolsure_averseness is None:

            if full_results:
                self.inutility, self.discolsure_averseness, self.full_perf1 = _evaluate_individual(real_space, latent_space,
                                                                                  learning_techniques,
                                                                                  clustering_technique,
                                                                                  full_results)
            else:
                self.inutility, self.discolsure_averseness = _evaluate_individual(real_space, latent_space,
                                                                           learning_techniques, clustering_technique)



