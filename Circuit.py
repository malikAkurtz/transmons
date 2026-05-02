import numpy as np

from Graph import Multidigraph
from Branch import Capacitor, Inductor, JosephsonElement

class Circuit():
    def __init__(self, graph: Multidigraph):
        self.nodes    = graph.nodes
        self.branches = graph.branches
        self.s        = graph.s
        self.t        = graph.t

        self.active_nodes, self.passive_nodes  = None, None
        self.P = None
        self.capacitance_matrix, self.inv_inductance_matrix = None, None

    def _partition_nodes(self):
        self.active_nodes  = []
        self.passive_nodes = []

        for node in self.nodes:
            node_id = node._id

            # skip ground node
            if node_id == 0:
                continue

            for branch in self.branches:
                if node_id in self.active_nodes or node_id in self.passive_nodes:
                    continue
                if not node_id in [self.s[branch._id], self.t[branch._id]]:
                    continue

                # check if it is connected to an inductor
                if isinstance(branch, Inductor) or isinstance(branch, JosephsonElement):
                    self.active_nodes.append(node_id)
                else:
                    self.passive_nodes.append(node_id)


    def _build_matrices(self):
        self.capacitance_matrix    = np.zeros((self.P, self.P))
        self.inv_inductance_matrix = np.zeros((self.P, self.P))

        # populate off-diagonals
        for i in range(self.P):
            for j in range(self.P):
                # if diagonal, skip
                if i == j:
                    continue
                # check if theres a branch connecting these nodes
                for branch in self.branches:
                    if set([i, j]) == set([self.s[branch._id], self.t[branch._id]]):
                        if isinstance(branch, Capacitor):
                            self.capacitance_matrix[i][j] -= branch.capacitance
                        elif isinstance(branch, Inductor):
                            self.inv_inductance_matrix -= (1/branch.inductance)

        # populate diagonals
        for i in range(self.P):
            self.capacitance_matrix[i][i]    = -1 * np.sum(self.capacitance_matrix[i])
            self.inv_inductance_matrix[i][i] = -1 * np.sum(self.inv_inductance_matrix[i])

        # get rid of ground row/column
        self.capacitance_matrix = np.delete(self.capacitance_matrix, 0, axis=0)
        self.capacitance_matrix = np.delete(self.capacitance_matrix, 0, axis=1)

        self.inv_inductance_matrix = np.delete(self.inv_inductance_matrix, 0, axis=0)
        self.inv_inductance_matrix = np.delete(self.inv_inductance_matrix, 0, axis=1)

    def build(self):
        self._partition_nodes()
        self.P = len(self.active_nodes) + len(self.passive_nodes) + 1
        self._build_matrices()


    def __str__(self):
        rep = ""
        for branch in self.branches:
            rep += "-" * 20 + "\n"
            if isinstance(branch, Capacitor):
                type = "Capacitor"
            elif isinstance(branch, Inductor):
                type = "Linear Inductor"
            elif isinstance(branch, JosephsonElement):
                type = "Josephson Element"
            rep += f"Branch: {branch._id}, {type}" + "\n"
            rep += f"{self.s[branch._id]} --> {self.t[branch._id]}" + "\n"
            rep += "-" * 20 + "\n"
        return rep
