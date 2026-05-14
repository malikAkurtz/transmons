import numpy as np
from Operator import Operator
from utils import get_pauli_coefs

def get_L1(U_q: np.ndarray, basis):
        return 1 - np.sum(np.abs(get_pauli_coefs(U_q, basis=basis))**2)

def get_r(coefs: np.ndarray):
    return np.linalg.norm(coefs)

def get_delta(r: float):
    return 1 - r

def get_process_fidelity(U_q: Operator, U_target: Operator, basis: str):
    return np.abs( np.trace(U_target[basis].conjugate().T @ U_q[basis]) )**2 / 4

def get_average_gate_fidelity(process_fidelity: float, L1: float):
    fidelity = ( (2 * process_fidelity) + 1 - L1 ) / 3
    return fidelity.real