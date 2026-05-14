
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from Architectures import *
from Graph import *
from Circuit import *
from Quantize import *
from Wavefunction import *
from utils import *
from CrankNicolson import *
from config import *
from constants import *
from fidelity import *
from Matrices import *

def main():
    """Run the end-to-end transmon-driving simulation."""
    # ------------------------ Hyperparameters ------------------------
    # Charge basis size; states run from -(n-1)/2 to +(n-1)/2.  Must be
    # odd so the basis is symmetric around zero.
    n_charge = 101
    n_trunc  = 7
    
    # ---- Initialize the Architecture (Single or Multi) ----
    nodes, branches, source_dict, terminal_dict, transmons, external_flux_on, external_flux_off = build_multi_qubit()

    # ---- Create Master Graph Representation ----
    master_graph = Multidigraph(
        nodes=nodes,
        branches=branches,
        source_dict=source_dict,
        terminal_dict=terminal_dict,
    )
    
    # ---- Create Master Circuit ----
    master_circuit = Circuit(
        graph=master_graph
    )

    # ---- Partition Circuit Nodes and Build Capacitance/Inverse Inductance Matrices ----
    master_circuit.build()
    
    # ---- Extract SubCircuits ----
    sub_circuits = [t.circuit for t in transmons]
    
    # ---- Quantize Each Subcircuit to Produce Unperturbed Subsystems----
    subsystems = []
    
    for i in range(len(transmons)):
        subsystem = quantize(
            circuit=sub_circuits[k], 
            charging_energy=master_circuit.charging_energy_matrix[k][k],
            external_flux=external_flux_off[k], 
            n_charge=n_charge
            )
        
        subsystems.append(subsystem)
    
    num_subsystems = len(subsystems)
    
    # ---- Create the System from the Master Circuit and Subsystems----
    system = System(
        circuit=master_circuit,
        subsystems=subsystems,
        n_charge=n_charge,
        n_trunc=n_trunc
    )
    
    print("Capacitance Matrix [fF]:")
    print(master_circuit.capacitance_matrix * 1e15)
    
    # ---- Retrieve Charging Energies of Each Transmon Circuit ----
    charging_energies = np.array([master_circuit.charging_energy_matrix[k][k] for k in range(num_subsystems)])
    
    # ---- Retrieve Idling Transmon Josephson Energies (From DCSQUID) ----
    josephson_energies = np.array([DCSQUID.calculate_effective_josephson_energy(
        left_josephson_energy=t.dcsquid.left_josephson_energy,
        right_josephson_energy=t.dcsquid.right_josephson_energy,
        external_flux=external_flux_off[k]
    ) for (k, t) in enumerate(transmons)])

    print(f"EC = {charging_energies}  EJ = {josephson_energies}  EJ/EC = {josephson_energies/charging_energies}")
    
    print("Idling Hamiltonian: ")
    print(system.H0["energy"])
    
    print("System Energies: ")
    print(system.energies)
    
    print("System Energy States: ")
    print(system.energy_states)
    
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
    
    if num_subsystems == 1:
        ground_col = 0          # |0>
        excited_col = 1         # |1>
    else:
        ground_col = 0          # |00>
        if k == 0:
            excited_col = 2     # |10>
        else:
            excited_col = 1     # |01>

    ground_state = Wavefunction(basis_to_coefs={"energy": system.logical_basis[:, ground_col]})
    excited_state = Wavefunction(basis_to_coefs={"energy": system.logical_basis[:, excited_col]})
    
    # ---- Initialize Solver ----
    solver = CrankNicolsonSolver()

    # # To use the SFQ lookup table instead of Gaussians, uncomment:
    # # lookup = pd.read_csv("sfq_V_lookup.csv")
    # # time, external_voltage = lookup["time_s"].to_numpy(), lookup["voltage_V"].to_numpy()

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
        initial_state=ground_state.copy(),
        external_voltage=voltage_full,
        time=time_full,
        k=k
    )
    
    plt.plot(time_full, P0, label='P0')
    plt.plot(time_full, P1, label='P1')
    plt.plot(time_full, P2, label='P2')
    plt.legend()
    plt.savefig("full_evolution.png")

    # ---- Extract Rabi Period ----
    if num_subsystems == 1:
        P_excited = P1  # column 1 = |1>
    else:
        if k == 0:
            P_excited = P2  # column 2 = |10>
        else:
            P_excited = P1  # column 1 = |01>

    P_smooth = uniform_filter1d(P_excited, size=STEPS_PER_PERIOD)
    peak_indices, _ = find_peaks(P_smooth, prominence=0.1)
    rabi_half_period = time_full[peak_indices[0]]
    
    # ---- Truncate from originals ----
    mask = time_full <= (rabi_half_period)
    time_half = time_full[mask]
    voltage_half = voltage_full[mask]

    # ---- Second evolution (half Rabi) ----
    final_state, P0, P1, P2 = solver.solve(
        system=system,
        initial_state=ground_state.copy(),
        external_voltage=voltage_half,
        time=time_half,
        k=k
    )
    
    plt.clf()
    plt.plot(time_half, P0, label='P0')
    plt.plot(time_half, P1, label='P1')
    plt.plot(time_half, P2, label='P2')
    plt.legend()
    plt.savefig("half_rabi_evolution.png")
    
    # ---- Compute the Effective Unitary On the Truncated Hilbert Space ----
    U = np.zeros((system.n_full, 2), dtype=complex)

    ground_evolved, _, _, _ = solver.solve(
        system=system,
        initial_state=ground_state.copy(),
        external_voltage=voltage_half,
        time=time_half,
        k=k
    )

    excited_evolved, _, _, _ = solver.solve(
        system=system,
        initial_state=excited_state.copy(),
        external_voltage=voltage_half,
        time=time_half,
        k=k
    )
    
    U[:, 0] = ground_evolved["energy"]
    U[:, 1] = excited_evolved["energy"]
    
    # ---- Project U Onto the Logical/Computational Subspace of Qubit k----
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
    



if __name__ == "__main__":
    main()
