"""
Node primitives for the circuit graph.

A :class:`Node` represents a single electrical node (a point of common voltage)
in a lumped-element circuit. Nodes are addressed by an auto-incrementing
integer id (``_id``) so that branches, source/target maps, and matrix indices
can refer to them unambiguously.

Convention
----------
The first :class:`Node` created in a session is treated as the *ground* node
(``_id == 0``) by the rest of the pipeline (e.g. :class:`Circuit`).  Build
your circuits by instantiating ground first, before any other nodes.
"""


class Node():
    """A single electrical node in the circuit graph.

    Attributes
    ----------
    _id : int
        Unique, monotonically increasing identifier assigned at construction
        time.  ``_id == 0`` is reserved for ground (the first node created).
    """

    # Class-level counter that hands out the next unique node id.
    global_id = 0

    def __init__(self):
        self._id = Node.global_id
        Node.global_id += 1
