import numpy as np
from Operator import Operator

I = np.eye(2)

X = np.array([
    [0, 1],
    [1, 0]
])

Y = np.array([
    [0, -1j],
    [1j, 0]
])

Z = np.array([
    [1, 0],
    [0, -1]
])

CZ = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -1.0]
])

PAULI_MATRICES = [I, X, Y, Z]

def get_RY_target(theta_target: float):
    RY_TARGET = Operator(
        basis_to_matrix={"energy": np.array([
                [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
                [np.sin(theta_target / 2), np.cos(theta_target / 2)]
            ])}
    )
    return RY_TARGET

def get_RX_target(theta_target: float):
    RX_TARGET = Operator(
        basis_to_matrix={"energy": np.array([
                [np.cos(theta_target / 2), -1j * np.sin(theta_target / 2)],
                [-1j * np.sin(theta_target / 2), np.cos(theta_target / 2)]
            ])}
    )
    return RX_TARGET