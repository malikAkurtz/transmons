"""
Branch primitives for the circuit graph.

A *branch* is a two-terminal lumped element that connects two
:class:`~Node.Node` objects in the circuit.  Three concrete element types are
supported:

* :class:`Capacitor`         — linear capacitor with capacitance ``C`` [F]
* :class:`Inductor`          — linear inductor  with inductance  ``L`` [H]
* :class:`JosephsonElement`  — non-linear Josephson element parameterised by
  its Josephson energy ``E_J`` [J]

Every branch is given a globally unique integer id (``_id``).  The id is used
as an index into the source/target arrays of the enclosing
:class:`~Graph.Multidigraph`, so branches must be created via the subclasses
below rather than the bare :class:`Branch` base.
"""


class Branch():
    """Abstract base class for all circuit branches.

    Notes
    -----
    Concrete subclasses are expected to assign ``self._id`` from the
    class-level ``global_id`` counter and to increment it.  The base class
    itself does not allocate an id (it is only used as a type marker and
    common ancestor).
    """

    # Class-level counter shared across every Branch subclass; used to
    # produce unique ids for every element added to a circuit.
    global_id = 0

    def __init__(self):
        pass


class Capacitor(Branch):
    """Linear capacitor branch.

    Parameters
    ----------
    capacitance : float
        Capacitance in Farads.
    """

    def __init__(self, capacitance: float):
        self._id = Branch.global_id
        Branch.global_id += 1

        self.capacitance = capacitance


class Inductor(Branch):
    """Linear (geometric) inductor branch.

    Parameters
    ----------
    inductance : float
        Inductance in Henries.
    """

    def __init__(self, inductance: float):
        self._id = Branch.global_id
        Branch.global_id += 1

        self.inductance = inductance


class JosephsonElement(Branch):
    """Non-linear Josephson element.

    Models the cosine potential of a Josephson junction; the geometric
    capacitance of the junction (if any) is represented separately by a
    sibling :class:`Capacitor` branch on the same pair of nodes.

    Parameters
    ----------
    josephson_energy : float
        Josephson energy ``E_J`` in Joules.
    """

    def __init__(self, josephson_energy: float):
        self._id = Branch.global_id
        Branch.global_id += 1

        self.josephson_energy = josephson_energy
