"""
Graph container used to describe a circuit topology.

A circuit is encoded as a directed multigraph: each branch has an explicit
*source* node and *target* node, and multiple branches are allowed between the
same pair of nodes (e.g. a Josephson junction is represented as a
:class:`~Branch.Capacitor` *and* a :class:`~Branch.JosephsonElement` sharing
the same endpoints).

The :class:`Multidigraph` class is intentionally a thin, immutable bundle of
the four arrays needed by :class:`~Circuit.Circuit` to assemble the
capacitance and inverse-inductance matrices.
"""

from Node import Node
from Branch import Branch


class Multidigraph():
    """Container describing the topology of a lumped-element circuit.

    Parameters
    ----------
    nodes : list[Node]
        All nodes in the circuit, including ground.  By convention the
        ground node is the one with ``_id == 0``.
    branches : list[Branch]
        All two-terminal elements (capacitors, inductors, Josephson
        elements) that connect the nodes.
    s : list[int]
        Source-node id for each branch.  ``s[i]`` is the id of the node at
        the source end of ``branches[i]``.
    t : list[int]
        Target-node id for each branch (same indexing convention as ``s``).

    Notes
    -----
    The (s, t) lists encode an *oriented* multigraph, but the matrix
    construction in :class:`~Circuit.Circuit` is symmetric under
    source/target swap because capacitance and inverse-inductance are
    symmetric two-terminal quantities.
    """

    def __init__(self, nodes: list[Node], branches: list[Branch], s: callable, t: callable):
        self.nodes = nodes
        self.branches = branches
        self.s = s
        self.t = t
