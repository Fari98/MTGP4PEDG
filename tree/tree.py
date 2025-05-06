from tree.utils import flatten, tree_depth, _execute_tree
import torch

class Tree:
    """
    The Tree class representing the candidate solutions in genetic programming.

    Attributes
    ----------
    repr_ : tuple or str
        Representation of the tree structure.
    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.
    depth : int
        Depth of the tree.
    fitness : float
        Fitness value of the tree.
    test_fitness : float
        Test fitness value of the tree.
    node_count : int
        Number of nodes in the tree.
    """

    TERMINALS = None
    FUNCTIONS = None
    CONSTANTS = None

    def __init__(self, repr_):
        """
        Initializes a Tree object.

        Parameters
        ----------
        repr_ : tuple
            Representation of the tree structure.
        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        self.repr_ = repr_
        self.depth = tree_depth(Tree.FUNCTIONS)(repr_)
        self.fitness = None
        # self.test_fitness = None
        self.size = len(list(flatten(self.repr_)))

    def predict(self, X):
        """
        Predict the tree semantics (output) for the given input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data to predict.

        Returns
        -------
        torch.Tensor
            The predicted output for the input data.

        Notes
        -----
        This function delegates the actual prediction task to the `apply_tree` method,
        which is assumed to be another method in the same class. The `apply_tree` method
        should be defined to handle the specifics of how predictions are made based on
        the tree structure used in this model.
        """
        return _execute_tree(
            repr_=self.repr_,
            X=X,
            FUNCTIONS=self.FUNCTIONS,
            TERMINALS=self.TERMINALS,
            CONSTANTS=self.CONSTANTS
        )

    def get_tree_representation(self, indent=""):
        """
        Returns the tree representation as a string with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.

        Returns
        -------
        str
            Returns the tree representation with the chosen indentation.
        """
        representation = []

        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            representation.append(indent + f"{function_name}(\n")

            # if the function has an arity of 2, process both left and right subtrees
            if Tree.FUNCTIONS[function_name]["arity"] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                representation.append(Tree(left_subtree).get_tree_representation(indent + "  "))
                representation.append(Tree(right_subtree).get_tree_representation(indent + "  "))
            # if the function has an arity of 1, process the left subtree
            else:
                left_subtree = self.repr_[1]
                representation.append(Tree(left_subtree).get_tree_representation(indent + "  "))

            representation.append(indent + ")\n")
        else:  # If it's a terminal node
            representation.append(indent + f"{self.repr_}\n")

        return "".join(representation)

    def print_tree_representation(self, indent=""):
        """
        Prints the tree representation with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.

        Returns
        -------
        None
            Prints the tree representation as a string with indentation.

        """

        print(self.get_tree_representation(indent=indent))