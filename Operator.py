"""
Multi-basis quantum operator container.

Quantum operators in this codebase are represented as dense matrices.  An
operator is often most naturally written in one basis (e.g. the charge basis
for a transmon Hamiltonian) but used in another (e.g. the energy
eigenbasis after diagonalisation).  :class:`Operator` stores the matrix
representation in *each* basis it has been computed for, keyed by a string
label such as ``"charge"`` or ``"energy"``.

This avoids re-computing basis transformations on the fly and lets callers
mix operators across bases with explicit, readable indexing
(``op["energy"]``).
"""

from __future__ import annotations

import numpy as np


class Operator():
    """A quantum operator stored as one or more matrix representations.

    Parameters
    ----------
    basis_to_matrix : dict[str, np.ndarray]
        Mapping from a basis label (e.g. ``"charge"``, ``"energy"``) to the
        matrix representation of the operator in that basis.  Both
        representations must describe the *same* underlying operator (i.e.
        related by a unitary basis transformation).

    Examples
    --------
    >>> n = Operator({"charge": np.diag([-1, 0, 1])})
    >>> n["charge"].shape
    (3, 3)
    >>> n["energy"] = U.conj().T @ n["charge"] @ U  # add a new basis
    """

    def __init__(self, basis_to_matrix: dict[str, np.ndarray]) -> None:
        self._basis_to_matrix = basis_to_matrix

    def __getitem__(self, basis: str) -> np.ndarray:
        """Return the matrix representation in the given basis."""
        return self._basis_to_matrix[basis]

    def __setitem__(self, basis: str, matrix: np.ndarray) -> None:
        """Store a new basis representation of the operator."""
        self._basis_to_matrix[basis] = matrix
        
    def __add__(self, other: Operator):
        new_op = Operator({})
        
        for basis, matrix in self._basis_to_matrix.items():
            new_op[basis] = self[basis] + other[basis]
        
        return new_op
    
    def __mul__(self, other: Operator | float):
        new_op = Operator({})
        
        if isinstance(other, Operator):
            for basis, matrix in self._basis_to_matrix.items():
                new_op[basis] = self[basis] @ other[basis]
        else:
            for basis, matrix in self._basis_to_matrix.items():
                new_op[basis] = self[basis] * other
        
        return new_op
        
        
            
