"""
Wavefunction container that mirrors the multi-basis design of
:class:`~Operator.Operator`.

A :class:`Wavefunction` keeps an array of complex amplitudes for each basis
it is expressed in (``"energy"``, ``"charge"``, …).  Each basis also carries
an associated unitary :class:`~Operator.Operator` ``U`` that records the
transformation from the original basis (initialised to identity), allowing
sequential operator applications to be tracked.
"""

import numpy as np

from Operator import Operator


class Wavefunction():
    """A quantum state expressed as amplitude vectors in one or more bases.

    Parameters
    ----------
    basis_to_coefs : dict[str, np.ndarray]
        Mapping from basis label to the complex coefficient vector of the
        state in that basis.

    Attributes
    ----------
    U : Operator
        Cumulative unitary transformations applied to the wavefunction in
        each basis.  Initialised to the identity for every basis present in
        ``basis_to_coefs``.
    """

    def __init__(self, basis_to_coefs: dict[str, np.ndarray]) -> None:
        self._basis_to_coefs = basis_to_coefs

        # Track the cumulative unitary transformation in each basis so that
        # callers can recover the propagator after an evolution loop.
        self.U = Operator({})
        for basis, coefs in self._basis_to_coefs.items():
            n = len(coefs)
            self.U[basis] = np.eye(n)

    def __getitem__(self, basis):
        """Return the coefficient vector in the requested basis."""
        return self._basis_to_coefs[basis]

    def __setitem__(self, basis, coefs) -> None:
        """Replace the coefficient vector in the given basis."""
        self._basis_to_coefs[basis] = coefs

    def apply(self, operator: Operator):
        """Apply ``operator`` to the wavefunction in every shared basis.

        For each basis present on this wavefunction, multiplies the
        coefficient vector by the matrix representation of ``operator`` in
        that basis.

        Parameters
        ----------
        operator : Operator
            Operator whose matrix in each basis acts on the corresponding
            coefficient vector.
        """
        for basis, coefs in self._basis_to_coefs:
            matrix = operator[basis]
            new_coefs = matrix @ coefs
            self[basis] = new_coefs
