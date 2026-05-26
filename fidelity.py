import numpy as np
from scipy.optimize import minimize
from utils import get_pauli_coefs

def get_L1(U_q: np.ndarray):
    pauli_coefs = get_pauli_coefs(U_q)
    return 1 - np.sum(np.abs(pauli_coefs)**2)

def get_lz_corrected_fidelity(U_q: np.ndarray, U_target: np.ndarray):
    """Maximum process fidelity over single-qubit Z rotations applied to U_q.

    Returns (F_pro, alphas) where alphas[i] is the optimal Z-rotation angle
    on qubit i. This is the "useful" fidelity — virtual-Z calibration on a
    real device absorbs these single-qubit phases for free.
    """
    d = len(U_q)
    n = int(np.log2(d))

    def rz(theta):
        return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])

    def neg_fpro(angles):
        R = np.array([[1.0 + 0j]])
        for i in range(n):
            R = np.kron(R, rz(angles[i]))
        U_corrected = R @ U_q
        return -np.abs(np.trace(U_target.conj().T @ U_corrected))**2 / d**2

    best = None
    rng = np.random.default_rng(0)
    for _ in range(8):
        x0 = rng.uniform(-np.pi, np.pi, n)
        res = minimize(neg_fpro, x0, method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return -best.fun, best.x

def get_r(coefs: np.ndarray):
    return np.linalg.norm(coefs)

def get_delta(r: float):
    return 1 - r

def get_process_fidelity(U_q: np.ndarray, U_target: np.ndarray):
    d = len(U_q)
    return np.abs( np.trace(U_target.conjugate().T @ U_q) )**2 / (d**2)

def get_average_gate_fidelity(U_q: np.ndarray, process_fidelity: float, L1: float):
    d = len(U_q)
    fidelity = ( (d * process_fidelity) + 1 - L1 ) / (d + 1)
    return fidelity.real