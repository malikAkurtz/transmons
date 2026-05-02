import numpy as np

from Circuit import Circuit
from Node import Node
from Branch import Capacitor, Inductor, JosephsonElement
from Graph import Multidigraph
from constants import REDUCED_FLUX_QUANTUM

class DCSQUID(Circuit):
    def __init__(self,
                 ground_node: Node,
                 left_jj_capacitance: float,
                 left_josephson_energy: float,
                 right_jj_capacitance: float,
                 right_josephson_energy: float):

        self.left_jj_capacitance    = left_jj_capacitance
        self.left_josephson_energy  = left_josephson_energy
        self.right_jj_capacitance   = right_jj_capacitance
        self.right_josephson_energy = right_josephson_energy

        self.gnd = ground_node

        # Create the island
        self.island = Node()

        self.nodes = [self.gnd, self.island]

        # Create the left Josephson Junction
        left_jj_capacitor      = Capacitor(self.left_jj_capacitance)
        left_josephson_element = JosephsonElement(self.left_josephson_energy)

        # Create the right Josephson Junction
        right_jj_capacitor      = Capacitor(self.right_jj_capacitance)
        right_josephson_element = JosephsonElement(self.right_josephson_energy)

        # Create list of branchs
        self.branches = [left_jj_capacitor, left_josephson_element,
                         right_jj_capacitor, right_josephson_element]

        # Create s and t maps, every source node is ground, every terminal node is the island
        self.s = [self.gnd._id] * len(self.branches)
        self.t = [self.island._id] * len(self.branches)

        graph = Multidigraph(
            nodes=self.nodes,
            branches=self.branches,
            s=self.s,
            t=self.t
        )

        super().__init__(graph)

    @staticmethod
    def calculate_effective_josephson_energy(left_josephson_energy: float, right_josephson_energy: float, external_flux: float):
        total_josephson_energy = left_josephson_energy + right_josephson_energy
        d = (right_josephson_energy - left_josephson_energy) / total_josephson_energy
        return total_josephson_energy \
                * np.cos(external_flux / (2 * REDUCED_FLUX_QUANTUM)) \
                    * np.sqrt(1 + (d**2) * np.tan(external_flux / (2 * REDUCED_FLUX_QUANTUM))**2)
