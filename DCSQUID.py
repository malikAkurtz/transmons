"""
DC SQUID circuit model.

A *DC SQUID* (Direct-Current Superconducting QUantum Interference Device) is a
loop containing two Josephson junctions in parallel, threaded by an external
flux.  In a transmon, it acts as a flux-tunable Josephson element: the two
junctions interfere to produce an *effective* Josephson energy that depends
on the applied flux.

The :class:`DCSQUID` class assembles the four constituent branches (two
Josephson elements and their two parallel geometric capacitances), wires them
between ground and a single *island* node, and forwards the topology to the
:class:`~Circuit.Circuit` base class.

The static method :meth:`DCSQUID.calculate_effective_josephson_energy`
returns the flux-dependent effective Josephson energy used by the
:mod:`Quantize` step.
"""

import numpy as np

from Circuit import Circuit
from Node import Node
from Branch import Capacitor, Inductor, JosephsonElement
from Graph import Multidigraph
from constants import REDUCED_FLUX_QUANTUM


class DCSQUID(Circuit):
    """Two Josephson junctions in parallel, possibly asymmetric.

    Parameters
    ----------
    ground_node : Node
        Pre-existing ground node (must have ``_id == 0``); the SQUID island
        is created internally and connected to ``ground_node`` by every
        branch.
    left_jj_capacitance, right_jj_capacitance : float
        Geometric (shunt) capacitance of each junction, in Farads.  Pass
        ``0`` if the junction's own capacitance is being absorbed into a
        separate shunt elsewhere in the circuit.
    left_josephson_energy, right_josephson_energy : float
        Josephson energy of each junction, in Joules.  The asymmetry
        ``d = (E_R - E_L) / (E_R + E_L)`` enters the effective-energy
        formula below.

    Notes
    -----
    The SQUID island is created during construction and exposed as
    ``self.island``; this is the node carrying the flux-degree of freedom
    of the transmon when the SQUID is wrapped by :class:`~Transmon.Transmon`.
    """

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

        # The single floating "island" node of the SQUID; every branch
        # below connects ground -> island.
        self.island = Node()

        self.nodes = [self.gnd, self.island]

        # Left junction: a Josephson element in parallel with its geometric
        # capacitance.
        left_jj_capacitor      = Capacitor(self.left_jj_capacitance)
        left_josephson_element = JosephsonElement(self.left_josephson_energy)

        # Right junction: same structure as the left.
        right_jj_capacitor      = Capacitor(self.right_jj_capacitance)
        right_josephson_element = JosephsonElement(self.right_josephson_energy)

        self.branches = [left_jj_capacitor, left_josephson_element,
                         right_jj_capacitor, right_josephson_element]

        # All branches share the same orientation: ground -> island.
        self.source_dict   = {branch: self.gnd for branch in self.branches}
        self.terminal_dict = {branch: self.island for branch in self.branches}

        self.graph = Multidigraph(
            nodes=self.nodes,
            branches=self.branches,
            source_dict=self.source_dict,
            terminal_dict=self.terminal_dict
        )

        super().__init__(self.graph)

    @staticmethod
    def calculate_effective_josephson_energy(left_josephson_energy: float, right_josephson_energy: float, external_flux: float):
        r"""Effective Josephson energy of the (possibly asymmetric) SQUID.

        For a SQUID with junction energies ``E_{J,L}`` and ``E_{J,R}`` and
        external flux ``\Phi_{ext}`` threading the loop, the effective
        Josephson energy seen by the transmon island is

        .. math::

            E_J^{\text{eff}}(\Phi_{ext}) = E_{J\Sigma}\,
                \cos\!\Big(\tfrac{\Phi_{ext}}{2\,\Phi_0^{*}}\Big)
                \sqrt{1 + d^2 \tan^2\!\Big(\tfrac{\Phi_{ext}}{2\,\Phi_0^{*}}\Big)}

        where ``E_{J\Sigma} = E_{J,L} + E_{J,R}`` is the sum,
        ``d = (E_{J,R} - E_{J,L}) / E_{J\Sigma}`` is the asymmetry, and
        ``\Phi_0^{*} = \hbar / (2e)`` is the *reduced* flux quantum.

        Parameters
        ----------
        left_josephson_energy, right_josephson_energy : float
            Josephson energies of the two junctions, in Joules.
        external_flux : float
            External flux threading the SQUID loop, in Webers.  Use
            multiples of :data:`constants.REDUCED_FLUX_QUANTUM` for
            convenience.

        Returns
        -------
        float
            Effective Josephson energy in Joules.
        """
        total_josephson_energy = left_josephson_energy + right_josephson_energy
        d = (right_josephson_energy - left_josephson_energy) / total_josephson_energy
        return total_josephson_energy \
                * np.cos(external_flux / (2 * REDUCED_FLUX_QUANTUM)) \
                    * np.sqrt(1 + (d**2) * np.tan(external_flux / (2 * REDUCED_FLUX_QUANTUM))**2)
