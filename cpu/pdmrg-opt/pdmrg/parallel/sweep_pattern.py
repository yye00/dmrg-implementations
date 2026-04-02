"""Staggered sweep pattern for PDMRG.

Even ranks (0, 2, 4...): start at RIGHT end of block, sweep LEFT first.
Odd ranks (1, 3, 5...): start at LEFT end of block, sweep RIGHT first.

This ensures processors meet at shared boundaries simultaneously.
"""


def get_initial_direction(rank):
    """Get the starting sweep direction for a given rank.

    Parameters
    ----------
    rank : int

    Returns
    -------
    direction : str
        'left' for even ranks, 'right' for odd ranks.
    """
    return 'left' if rank % 2 == 0 else 'right'


def get_initial_position(rank, n_local):
    """Get the starting local site index for a given rank.

    Even ranks start at the RIGHT end (last site).
    Odd ranks start at the LEFT end (first site).

    Parameters
    ----------
    rank : int
    n_local : int
        Number of local sites.

    Returns
    -------
    position : int
        Local site index to start at.
    """
    if rank % 2 == 0:
        return n_local - 1  # Right end
    else:
        return 0  # Left end
