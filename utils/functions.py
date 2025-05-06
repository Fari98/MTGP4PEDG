import torch

def protected_div(x1, x2):
    """ Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    return torch.where(torch.abs(x2) > 0.001, torch.div(x1, x2), torch.tensor(1.0, dtype=x2.dtype, device=x2.device))


def mean_(x1, x2):

    return torch.div(torch.add(x1, x2), 2)

FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    'mean': {'function': lambda x, y: mean_(x, y), 'arity': 2},
    'tan': {'function': lambda x: torch.tan(x), 'arity': 1},
    'sin': {'function': lambda x: torch.sin(x), 'arity': 1},
    'cos': {'function': lambda x: torch.cos(x), 'arity': 1},
}