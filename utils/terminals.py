def get_terminals(X):
    """
    Get terminal nodes for a dataset.

    Parameters
    ----------
    X : (torch.Tensor)
        An array to get the set of TERMINALS from, it will correspond to the columns.

    Returns
    -------
    dict
        Dictionary of terminal nodes.
    """

    return  {f"x{i}": i for i in range(len(X[0]))}