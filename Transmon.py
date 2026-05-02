import numpy as np

from Circuit import Circuit
from DCSQUID import DCSQUID
from Branch import Capacitor
from Graph import Multidigraph

class Transmon(Circuit):
    def __init__(self, dcsquid: DCSQUID, shunt_capacitance: float, coupling_capacitance: float):

        self.shunt_capacitance    = shunt_capacitance
        self.coupling_capacitance = coupling_capacitance

        self.dcsquid = dcsquid

        # Create the shunt capacitor
        shunt_capacitor = Capacitor(self.shunt_capacitance)

        # Create the coupling capacitor
        coupling_capacitor = Capacitor(self.coupling_capacitance)

        graph = Multidigraph(
            nodes=dcsquid.nodes,
            branches=dcsquid.branches + [shunt_capacitor] + [coupling_capacitor],
            s=dcsquid.s + [self.dcsquid.gnd._id] * 2,
            t=dcsquid.t + [self.dcsquid.island._id] * 2
        )

        super().__init__(graph)
