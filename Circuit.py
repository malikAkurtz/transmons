"""
Generic lumped-element circuit assembled from a :class:`~Graph.Multidigraph`.

Given a topology (nodes + branches + source/target maps), :class:`Circuit`
builds the two reduced linear-algebra objects needed for canonical
quantisation of LC-like circuits:

* the **reduced capacitance matrix** ``C``  (with the ground row/column
  removed), and
* the **reduced inverse-inductance matrix** ``L^{-1}``.

The Josephson cosine potential is *not* part of the linear matrices; it is
applied later during quantisation (see :mod:`Quantize`).

Node partitioning
-----------------
Nodes are split into:

* *active* nodes — connected to at least one inductive element
  (:class:`~Branch.Inductor` or :class:`~Branch.JosephsonElement`).  These
  carry a flux degree of freedom.
* *passive* nodes — connected only to capacitors.  These can in principle
  be eliminated by Schur complement (the current code keeps them in the
  matrix and lets the quantisation step handle the size).

The ground node (id ``0``) is always present and is removed from the final
matrices.
"""

import numpy as np

from Graph import Multidigraph
from Branch import Capacitor, Inductor, JosephsonElement


class Circuit():
    """A lumped-element circuit built from a topology graph.

    Parameters
    ----------
    graph : Multidigraph
        Topology of the circuit (nodes, branches, and source/target maps).

    Attributes
    ----------
    nodes, branches, s, t
        Mirrors of the corresponding fields on ``graph``.
    active_nodes, passive_nodes : list[int]
        Populated by :meth:`build`; ids of inductively-connected nodes vs
        purely capacitive nodes (excluding ground).
    P : int
        Total node count including ground.  Set in :meth:`build`.
    capacitance_matrix : np.ndarray
        ``(P-1) x (P-1)`` reduced capacitance matrix in Farads, with the
        ground row/column removed.
    inv_inductance_matrix : np.ndarray
        ``(P-1) x (P-1)`` reduced inverse-inductance matrix in Henries^-1,
        with the ground row/column removed.
    """

    def __init__(self, graph: Multidigraph):
        self.nodes    = graph.nodes
        self.branches = graph.branches
        self.s        = graph.s
        self.t        = graph.t

        self.active_nodes, self.passive_nodes  = None, None
        self.P = None
        self.capacitance_matrix, self.inv_inductance_matrix = None, None

    def _partition_nodes(self):
        """Classify each non-ground node as *active* or *passive*.

        A node is *active* if at least one of its incident branches is an
        :class:`~Branch.Inductor` or :class:`~Branch.JosephsonElement`;
        otherwise it is *passive*.  Ground (id 0) is excluded.
        """
        self.active_nodes  = []
        self.passive_nodes = []

        for node in self.nodes:
            node_id = node._id

            # Skip ground; it is treated separately and removed at the end.
            if node_id == 0:
                continue

            for branch in self.branches:
                # A node is classified the first time we encounter an
                # incident branch; later branches on the same node are
                # ignored.
                if node_id in self.active_nodes or node_id in self.passive_nodes:
                    continue
                if not node_id in [self.s[branch._id], self.t[branch._id]]:
                    continue

                # A node attached to *any* inductive element is active.
                if isinstance(branch, Inductor) or isinstance(branch, JosephsonElement):
                    self.active_nodes.append(node_id)
                else:
                    self.passive_nodes.append(node_id)


    def _build_matrices(self):
        """Assemble the reduced capacitance and inverse-inductance matrices.

        Implements the standard nodal construction:

        * Off-diagonal ``[i, j]`` (with ``i != j``) is the *negative* sum
          of element values for every branch connecting node ``i`` to node
          ``j``.
        * Diagonal ``[i, i]`` is set so that each row sums to zero before
          ground elimination (Kirchhoff's law at the node).

        The ground row and column are deleted at the end.
        """
        self.capacitance_matrix    = np.zeros((self.P, self.P))
        self.inv_inductance_matrix = np.zeros((self.P, self.P))

        # Off-diagonal entries: accumulate -C and -1/L for every branch
        # whose endpoints are {i, j}.
        for i in range(self.P):
            for j in range(self.P):
                if i == j:
                    continue
                for branch in self.branches:
                    if set([i, j]) == set([self.s[branch._id], self.t[branch._id]]):
                        if isinstance(branch, Capacitor):
                            self.capacitance_matrix[i][j] -= branch.capacitance
                        elif isinstance(branch, Inductor):
                            self.inv_inductance_matrix -= (1/branch.inductance)

        # Diagonal entries: enforce row-sum = 0 (i.e. each node's self-term
        # equals the sum of magnitudes of its connections).
        for i in range(self.P):
            self.capacitance_matrix[i][i]    = -1 * np.sum(self.capacitance_matrix[i])
            self.inv_inductance_matrix[i][i] = -1 * np.sum(self.inv_inductance_matrix[i])

        # Drop the ground (row/column 0) to obtain the reduced matrices.
        self.capacitance_matrix = np.delete(self.capacitance_matrix, 0, axis=0)
        self.capacitance_matrix = np.delete(self.capacitance_matrix, 0, axis=1)

        self.inv_inductance_matrix = np.delete(self.inv_inductance_matrix, 0, axis=0)
        self.inv_inductance_matrix = np.delete(self.inv_inductance_matrix, 0, axis=1)

    def build(self):
        """Run node partitioning and matrix construction.

        Must be called before :attr:`capacitance_matrix` or
        :attr:`inv_inductance_matrix` are read.
        """
        self._partition_nodes()
        self.P = len(self.active_nodes) + len(self.passive_nodes) + 1
        self._build_matrices()
        
    def hamiltonian(self):
        pass


    def __str__(self):
        """Human-readable list of the branches in the circuit.

        Each branch is shown with its id, element type, and oriented
        endpoints ``source --> target``.
        """
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
