"""
Transmon circuit: a flux-tunable DC SQUID shunted by a large capacitor.

The :class:`Transmon` wraps a :class:`~DCSQUID.DCSQUID` and adds two
extra capacitors between the SQUID island and ground:

* a large **shunt capacitance** ``C_s`` that suppresses charge dispersion
  and pushes the qubit into the transmon regime ``E_J / E_C >> 1``, and
* a small **coupling capacitance** ``C_g`` that lets an external voltage
  drive the island.

The combined topology is forwarded to :class:`~Circuit.Circuit`, which
constructs the reduced capacitance matrix used downstream by
:func:`Quantize.quantize_transmon`.
"""

import numpy as np

from Circuit import Circuit
from DCSQUID import DCSQUID
from Branch import Capacitor
from Graph import Multidigraph
from constants import e
from Expression import *


class Transmon(Circuit):
    """Transmon qubit built on top of an existing DC SQUID.

    Parameters
    ----------
    dcsquid : DCSQUID
        Pre-built SQUID providing the non-linear inductive element and the
        ground/island nodes.
    shunt_capacitance : float
        Shunt capacitance ``C_s`` between island and ground, in Farads.
        This is typically the dominant capacitance in the circuit and sets
        the charging energy ``E_C = e^2 / (2 C_\Sigma)``.
    coupling_capacitance : float
        Coupling capacitance ``C_g`` between island and ground (representing
        coupling to an external drive line), in Farads.  Used in the drive
        Hamiltonian as the coupling efficiency ``C_g / C_\Sigma``.
    """

    def __init__(self, dcsquid: DCSQUID, shunt_capacitance: float, coupling_capacitance: float):

        self.dcsquid = dcsquid
        self.gnd     = dcsquid.gnd
        self.island = self.dcsquid.island
                
        self.shunt_capacitance    = shunt_capacitance
        self.coupling_capacitance = coupling_capacitance

        # Extra capacitors that distinguish a transmon from a bare SQUID:
        # a big shunt to set E_C, and a small coupling capacitor that the
        # external drive enters through.
        shunt_capacitor = Capacitor(self.shunt_capacitance)
        coupling_capacitor = Capacitor(self.coupling_capacitance)
        
        self.branches = self.dcsquid.branches + [shunt_capacitor] + [coupling_capacitor]

        # Combined topology: the SQUID branches plus the two new shunt-
        # to-ground capacitors, all oriented ground -> island.
        self.source_dict   = {**self.dcsquid.source_dict, 
                              shunt_capacitor: self.gnd, 
                              coupling_capacitor: self.gnd
                              }
        self.terminal_dict = {**self.dcsquid.terminal_dict,
                              shunt_capacitor: self.island,
                              coupling_capacitor: self.island
                              }
        
        self.graph = Multidigraph(
            nodes=dcsquid.nodes,
            branches=self.branches,
            source_dict=self.source_dict,
            terminal_dict=self.terminal_dict
        )
                
        super().__init__(self.graph)
        
    def hamiltonian(self, charging_energy: float, external_flux: float):
        EJ = DCSQUID.calculate_effective_josephson_energy(
            left_josephson_energy=self.dcsquid.left_josephson_energy,
            right_josephson_energy=self.dcsquid.right_josephson_energy,
            external_flux=external_flux
        )
        return Scalar(4) * Scalar(charging_energy) * ( Op("n")**Scalar(2) ) + Scalar(-EJ) * Cos(Op("phi"))
