"""
Top-level simulation: drive a quantised transmon with an SFQ pulse train.

Pipeline
--------
1. **Build** the lumped-element circuit (DC SQUID + shunt + coupling).
2. **Quantise** it: form the charge-basis Hamiltonian, diagonalise, and
   transform the charge operator into the energy basis.
3. **Synthesise** a Gaussian SFQ pulse train (or load an RCSJ-derived one
   from ``sfq_V_lookup.csv``).
4. **Evolve** the qubit under ``H_0 + H_D(t)`` using the Crank–Nicolson
   solver.
5. **Plot** the lowest three energy-level populations vs time.

Run with::

    python main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import e, h, REDUCED_FLUX_QUANTUM
from config import NUM_KICKS, AMPLITUDE_SCALE, DETUNING, STEPS_PER_PERIOD, SIGMA
from Node import Node
from DCSQUID import DCSQUID
from Transmon import Transmon
from System import System
from Quantize import quantize
from CrankNicolson import CrankNicolsonSolver
from Wavefunction import Wavefunction
from utils import create_gaussian_sfq_pulses
from Graph import Multidigraph
from Circuit import Circuit
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from Matrices import get_RX_target, get_RY_target
from fidelity import *


def main():
    """Run the end-to-end transmon-driving simulation."""
    # ------------------------ Hyperparameters ------------------------
    # Charge basis size; states run from -(n-1)/2 to +(n-1)/2.  Must be
    # odd so the basis is symmetric around zero.
    n_charge = 101
    n_trunc  = 7
    
    # ---- Circuit Paramters ----
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
    
    # ---- Create Graph Representation ----
    graph = Multidigraph(
        nodes=nodes,
        branches=branches,
        source_dict=source_dict,
        terminal_dict=terminal_dict,
    )
    
    # ---- Create Multi-Qubit Transmon Circuit ----
    circuit = Circuit(
        graph=graph
    )

    # ---- Partition Circuit Nodes and Build Capacitance/Inverse Inductance Matrices ----
    circuit.build()
    
    # ---- Quantize the Circuit Object ----
    subsystems = [quantize(circuit=t, 
                           charging_energy=circuit.charging_energy_matrix[k][k],
                           external_flux=external_flux_off[k], n_charge=n_charge)
                  for (k, t) in enumerate(transmons)]
    
    # ---- Create the System from the SubSystems----
    system = System(
        circuit=circuit,
        subsystems=subsystems,
        n_charge=n_charge,
        n_trunc=n_trunc
    )
    
    print("Capacitance Matrix [fF]:")
    print(circuit.capacitance_matrix * 1e15)
    
    # ---- Retrieve Charging Energies of Each Transmon Circuit ----
    charging_energies = np.array([circuit.charging_energy_matrix[k][k] for k in range(len(subsystems))])
    
    # ---- Retrieve Idling Transmon Josephson Energies (From DCSQUID) ----
    josephson_energies = np.array([DCSQUID.calculate_effective_josephson_energy(
        left_josephson_energy=t.dcsquid.left_josephson_energy,
        right_josephson_energy=t.dcsquid.right_josephson_energy,
        external_flux=external_flux_off[k]
    ) for (k, t) in enumerate(transmons)])

    print(f"EC = {charging_energies}  EJ = {josephson_energies}  EJ/EC = {josephson_energies/charging_energies}")
    
    print("Idling Hamiltonian: ")
    print(system.H0["energy"])
    
    # ---- Qubit Frequencies ----
    print("Qubit Frequencies")
    print(system.frequencies)
    
    # ---- Qubit Angular Frequencies ----
    print("Qubit Angular Frequencies")
    print(system.angular_frequencies)
    
    # ---- Logical Basis ----
    print("Logical Basis")
    print(system.logical_basis)

    # ---- Qubit Subsystem Index to Drive ----
    k = 0
    
    # ---- Logical |0> ----
    z_amps = np.zeros(n_trunc**len(subsystems))
    z_amps[0] = 1
    z = Wavefunction(basis_to_coefs={"energy": z_amps})
    
    # ---- Logical |1> ----
    o_amps = np.zeros(n_trunc**len(subsystems))
    o_amps[1] = 1
    o  = Wavefunction(basis_to_coefs={"energy": o_amps})
    
    # ---- Initialize Solver ----
    solver        = CrankNicolsonSolver()

    # To use the SFQ lookup table instead of Gaussians, uncomment:
    # lookup = pd.read_csv("sfq_V_lookup.csv")
    # time, external_voltage = lookup["time_s"].to_numpy(), lookup["voltage_V"].to_numpy()

    time, external_voltage = create_gaussian_sfq_pulses(
        num_kicks=NUM_KICKS,
        amplitude_scale=AMPLITUDE_SCALE,
        driving_period=(1 / system.frequencies[k]) + DETUNING,
        pulse_width=SIGMA,
        steps_per_period=STEPS_PER_PERIOD
    )

    # ---- Save originals ----
    time_full = time.copy()
    voltage_full = external_voltage.copy()

    # ---- First evolution (full) ----
    final_state, P0, P1, P2 = solver.solve(
        system=system,
        initial_state=z.copy(),
        external_voltage=voltage_full,
        time=time_full,
        k=k
    )

    # ---- Extract Rabi Period ----
    P1_smooth = uniform_filter1d(P1, size=STEPS_PER_PERIOD)
    peak_indices, _ = find_peaks(P1_smooth, prominence=0.1)
    rabi_half_period = time_full[peak_indices[0]]
    print("Rabi Half-Period: ", rabi_half_period)
    
    # ---- Truncate from originals ----
    mask = time_full <= (rabi_half_period)
    time_half = time_full[mask]
    voltage_half = voltage_full[mask]

    # ---- Second evolution (half Rabi) ----
    final_state, P0, P1, P2 = solver.solve(
        system=system,
        initial_state=z.copy(),
        external_voltage=voltage_half,
        time=time_half,
        k=k
    )
    
    # ---- Compute the Effective Unitary On the Truncated Hilbert Space ----
    U = np.zeros((n_trunc, n_trunc), dtype=complex)
    
    z_evolved, _, _, _ = solver.solve(
        system=system,
        initial_state=z.copy(),
        external_voltage=voltage_half,
        time=time_half,
        k=k
    )
    
    o_evolved, _, _, _ = solver.solve(
        system=system,
        initial_state=o.copy(),
        external_voltage=voltage_half,
        time=time_half,
        k=k
    )
    
    U[:, 0] = z_evolved["energy"]
    U[:, 1] = o_evolved["energy"]
    
    # ---- Project U Onto the Computational Subspace ----
    U_q = Operator(basis_to_matrix={"energy": U[:2, :2]})
    
    # ---- Calculate Leakage and Fidelity Metrics ----
    pauli_coefs = get_pauli_coefs(
        U_q=U_q,
        basis="energy"
    )
    
    L1 = get_L1(
        U_q=U_q,
        basis="energy"
    )
    
    r = get_r(
        coefs=pauli_coefs,
    )
    
    process_fidelity = get_process_fidelity(
        U_q=U_q,
        U_target=get_RY_target(theta_target=np.pi),
        basis="energy"
    )

    fidelity = get_average_gate_fidelity(
        process_fidelity=process_fidelity,
        L1=L1
    )
    
    print("Gate Fidelity: ")
    print(fidelity)
    
    plt.plot(time_half, P0, label='P0')
    plt.plot(time_half, P1, label='P1')
    plt.plot(time_half, P2, label='P2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
