import numpy as np

from Operator import Operator

class Wavefunction():
    def __init__(self, basis_to_coefs: dict[str, np.ndarray]) -> None:
        self._basis_to_coefs = basis_to_coefs

        self.U = Operator({})
        for basis, coefs in self._basis_to_coefs.items():
            n = len(coefs)
            self.U[basis] = np.eye(n)

    def __getitem__(self, basis):
        return self._basis_to_coefs[basis]

    def __setitem__(self, basis, coefs) -> None:
        self._basis_to_coefs[basis] = coefs

    def apply(self, operator: Operator):
        for basis, coefs in self._basis_to_coefs:
            matrix = operator[basis]
            new_coefs = matrix @ coefs
            self[basis] = new_coefs
