import numpy as np

from System import System
from Wavefunction import Wavefunction
from constants import e, hbar

class CrankNicolsonSolver():
    def __init__(self):
        pass

    def solve(self, system: System, initial_state: Wavefunction, external_voltage: np.ndarray, time: float):

        n = len(initial_state["energy"])

        num_steps = len(external_voltage)

        dt = time[1] - time[0]

        state = initial_state

        P0 = np.zeros(num_steps)
        P1 = np.zeros(num_steps)
        P2 = np.zeros(num_steps)

        for i in range(num_steps - 1):

            voltage_midpoint = (external_voltage[i] + external_voltage[i+1]) / 2

            HD_midpoint = (2 * e) * (system.circuit.coupling_capacitance / system.circuit.capacitance_matrix[0][0]) * voltage_midpoint * system.n["energy"]

            H = system.H0["energy"] + HD_midpoint

            A = np.eye(n) + (1j * (dt / (2*hbar)) * H)
            B = np.eye(n) - (1j * (dt / (2*hbar)) * H)

            state["energy"] = np.linalg.solve(A, B @ state["energy"])

            P0[i] = np.abs(state["energy"][0])**2
            P1[i] = np.abs(state["energy"][1])**2
            P2[i] = np.abs(state["energy"][2])**2

        return state, P0, P1, P2
