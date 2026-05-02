import numpy as np

class Operator():
    def __init__(self, basis_to_matrix: dict[str, np.ndarray]) -> None:
        self._basis_to_matrix = basis_to_matrix

    def __getitem__(self, basis: str) -> np.ndarray:
        return self._basis_to_matrix[basis]

    def __setitem__(self, basis: str, matrix: np.ndarray) -> None:
        self._basis_to_matrix[basis] = matrix
