import numpy as np

from Circuit import Circuit
from Operator import Operator

class System():
    def __init__(self, circuit: Circuit, charge_operator: Operator, unperturbed_hamiltonian: Operator):
        self.circuit  = circuit
        self.n        = charge_operator
        self.H0       = unperturbed_hamiltonian
        self.state    = None
