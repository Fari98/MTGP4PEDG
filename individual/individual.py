from tree.tree import Tree
import torch
from individual.utils import _evaluate_individual


class Individual:

    def __init__(self, representations, utility = None, disclosure_averseness = None,
                                        # utility_test = None, disclosure_averseness_test = None #todo check wheter to include
                 ):


        self.trees = [Tree(tree) if not isinstance(tree, Tree) else tree for tree in representations]
        self.representations = [tree.repr_ if isinstance(tree, Tree) else tree for tree in representations]

        self.lenght = len(representations)

        self.utility = utility
        self.discolsure_averseness = disclosure_averseness

        # todo nodes depth information ??


    def predict(self, latent_space):

        return torch.concatenate([tree.predict(latent_space).unsqueeze(1) for tree in self.trees], dim=1)


    def evaluate(self, real_space, latent_space, learning_techniques, clustering_technique):

        if self.utility is None and self.discolsure_averseness is None:

           self.utility, self.discolsure_averseness = _evaluate_individual(real_space, latent_space,
                                                                           learning_techniques, clustering_technique)



