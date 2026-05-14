import numpy as np
from Operator import Operator
from utils import get_pauli_coefs, get_multi_qubit_paulis

def get_L1(U_q: np.ndarray):
    d = len(U_q)
    
    pauli_coefs = get_multi_qubit_paulis(num_qubits=int(np.log2(d)))
    
    return 1 - np.sum(np.abs(pauli_coefs)**2)

def get_r(coefs: np.ndarray):
    return np.linalg.norm(coefs)

def get_delta(r: float):
    return 1 - r

def get_process_fidelity(U_q: np.ndarray, U_target: np.ndarray):
    d = len(U_q)
    return np.abs( np.trace(U_target.conjugate().T @ U_q) )**2 / (d**2)

def get_average_gate_fidelity(process_fidelity: float, L1: float):
    fidelity = ( (2 * process_fidelity) + 1 - L1 ) / 3
    return fidelity.real