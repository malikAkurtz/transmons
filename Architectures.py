import numpy as np

from constants import REDUCED_FLUX_QUANTUM
from Node import Node
from DCSQUID import DCSQUID
from Transmon import Transmon
from Branch import Capacitor

def build_single_qubit():
    # ---- External Flux Threading JJ ----
    external_flux_on     = np.array([0.130]) * REDUCED_FLUX_QUANTUM
    external_flux_off    = np.array([0.130]) * REDUCED_FLUX_QUANTUM

    # SQUID + drive parameters.
    shunt_capacitance    = 70e-15   # [F]
    coupling_capacitance = 1e-15    # [F]

    # Junction parameters.  Asymmetric junctions give a flux-tunable but
    # finite minimum E_J.
    left_jj_capacitance    = 0
    left_jj_energy         = 7e-9  * REDUCED_FLUX_QUANTUM  # [J]
    right_jj_capacitance   = 0
    right_jj_energy        = 21e-9 * REDUCED_FLUX_QUANTUM  # [J]

    # ---- Build Ground Node ----
    gnd = Node()

    # ---- Build DCSQUID Circuit Object ----
    dcsquid = DCSQUID(
        ground_node=gnd,
        left_jj_capacitance=left_jj_capacitance,
        left_josephson_energy=left_jj_energy,
        right_jj_capacitance=right_jj_capacitance,
        right_josephson_energy=right_jj_energy
    )

    transmon = Transmon(
        dcsquid=dcsquid,
        shunt_capacitance=shunt_capacitance,
        coupling_capacitance=coupling_capacitance
    )
    
    transmons = [transmon]
        
    # ---- Create New Nodes List ----
    nodes = [gnd, transmon.island]
    
    # ---- Create New Branches List ----
    branches = transmon.branches
    
    # ---- Create New Source and Terminal Dicts ----
    source_dict = {**transmon.graph.source_dict}
    
    terminal_dict = {**transmon.graph.terminal_dict}
    
    return nodes, branches, source_dict, terminal_dict, transmons, external_flux_on, external_flux_off

def build_multi_qubit():
    # ---- External Flux Threading JJ ----
    external_flux_on     = np.array([0.130, 0.352, 0.130]) * REDUCED_FLUX_QUANTUM
    external_flux_off    = np.array([0.130, 0.376, 0.130]) * REDUCED_FLUX_QUANTUM
    
    # ---- Circuit Capacitances ----
    shunt_capacitance_1 = 70e-15   # [F]
    shunt_capacitance_2 = 70e-15   # [F]
    shunt_capacitance_c = 60e-15   # [F]
    
    coup_capacitance_12 = 0.25e-15 # [F]
    coup_capacitance_1c = 2e-15    # [F]
    coup_capacitance_2c = 2e-15    # [F]
    
    coup_capacitance_1e = 7.5e-15  # [F]
    coup_capacitance_2e = 7.5e-15  # [F]
    
    # ---- Circuit Josephson Energies ----
    left_jj_capacitance_1  = 0
    right_jj_capacitance_1 = 0
    left_jj_energy_1       = 7e-9  * REDUCED_FLUX_QUANTUM  # [J]
    right_jj_energy_1      = 21e-9 * REDUCED_FLUX_QUANTUM  # [J]
    
    left_jj_capacitance_c  = 0
    right_jj_capacitance_c = 0
    left_jj_energy_c       = 18e-9 * REDUCED_FLUX_QUANTUM  # [J]
    right_jj_energy_c      = 36e-9 * REDUCED_FLUX_QUANTUM  # [J]
    
    left_jj_capacitance_2  = 0
    right_jj_capacitance_2 = 0
    left_jj_energy_2       = 7e-9  * REDUCED_FLUX_QUANTUM  # [J]
    right_jj_energy_2      = 21e-9 * REDUCED_FLUX_QUANTUM  # [J]

    # ---- Build Ground Node ----
    gnd = Node()

    # ---- Build DCSQUID Circuit Objects ----
    dcsquid_1 = DCSQUID(
        ground_node=gnd,
        left_jj_capacitance=left_jj_capacitance_1,
        left_josephson_energy=left_jj_energy_1,
        right_jj_capacitance=right_jj_capacitance_1,
        right_josephson_energy=right_jj_energy_1
    )
    
    dcsquid_c = DCSQUID(
        ground_node=gnd,
        left_jj_capacitance=left_jj_capacitance_c,
        left_josephson_energy=left_jj_energy_c,
        right_jj_capacitance=right_jj_capacitance_c,
        right_josephson_energy=right_jj_energy_c
    )
    
    dcsquid_2 = DCSQUID(
        ground_node=gnd,
        left_jj_capacitance=left_jj_capacitance_2,
        left_josephson_energy=left_jj_energy_2,
        right_jj_capacitance=right_jj_capacitance_2,
        right_josephson_energy=right_jj_energy_2
    )
    
    # ---- Build Transmon Circuit Objects ----
    transmon_1 = Transmon(
        dcsquid=dcsquid_1,
        shunt_capacitance=shunt_capacitance_1,
        coupling_capacitance=coup_capacitance_1e
    )
    
    transmon_c = Transmon(
        dcsquid=dcsquid_c,
        shunt_capacitance=shunt_capacitance_c,
        coupling_capacitance=0
    )
    
    transmon_2 = Transmon(
        dcsquid=dcsquid_2,
        shunt_capacitance=shunt_capacitance_2,
        coupling_capacitance=coup_capacitance_2e
    )
    
    transmons = [transmon_1, transmon_c, transmon_2]
    
    # ---- Create Branches For Inter-Island Capacitances ----
    cap_12 = Capacitor(capacitance=coup_capacitance_12)
    
    cap_1c = Capacitor(capacitance=coup_capacitance_1c)
    
    cap_2c = Capacitor(capacitance=coup_capacitance_2c)
    
    # ---- Create New Nodes List ----
    nodes = [gnd, transmon_1.island, transmon_c.island, transmon_2.island]
    
    # ---- Create New Branches List ----
    branches = transmon_1.branches + transmon_c.branches + transmon_2.branches + [cap_12, cap_1c, cap_2c]
    
    # ---- Create New Source and Terminal Dicts ----
    source_dict = {**transmon_1.graph.source_dict, 
                   **transmon_c.graph.source_dict,
                   **transmon_2.graph.source_dict,
                   cap_12: transmon_1.island,
                   cap_1c: transmon_1.island,
                   cap_2c: transmon_2.island
                   }
    terminal_dict = {**transmon_1.graph.terminal_dict, 
                   **transmon_c.graph.terminal_dict,
                   **transmon_2.graph.terminal_dict,
                   cap_12: transmon_2.island,
                   cap_1c: transmon_c.island,
                   cap_2c: transmon_c.island
                   }

    return nodes, branches, source_dict, terminal_dict, transmons, external_flux_on, external_flux_off