
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
    n_trunc  = 5
    
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
    
    # ---- Quantize Each Subcircuit to Produce Unperturbed Subsystems----
    subsystems = []
    for k in range(len(transmons)):
        subsystem = quantize(
            circuit=transmons[k], 
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
    print("Qubit Frequencies: ")
    print(system.frequencies)
    
    # ---- Qubit Periods ----
    print("Qubit Periods: ")
    print(system.periods)
    
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
    
    # ---- Total Simulation Time ----
    # 70 nanoseconds
    total_time = 70e-9 # [s]
    
    # total periods
    total_periods = int(total_time / system.periods[k])
    
    # total steps
    num_steps = total_periods * STEPS_PER_PERIOD
    
    # delta t
    dt = total_time / num_steps
    
    time = np.arange(num_steps) * dt
    
    # ---- Create External Voltage Arrays (No External Driving) ----
    external_voltages = np.array([
        np.zeros(num_steps),
        np.zeros(num_steps),
        np.zeros(num_steps)
    ])
    
    # ---- Create Coupler Flux Schedule (raised-cosine ramp) ----
    coupler_flux_off = external_flux_off[1]
    coupler_flux_on  = external_flux_on[1]

    t_start = 10e-9   # ramp begins
    t_rise  = 5e-9    # rise / fall time
    t_hold  = 6e-9    # plateau at the ON flux (sweep this for calibration)

    coupler_flux_schedule = cosine_ramp_schedule(
        num_steps=num_steps,
        dt=dt,
        t_start=t_start,
        t_rise=t_rise,
        t_hold=t_hold,
        off_value=coupler_flux_off,
        on_value=coupler_flux_on,
    )

    # ---- Define Logical States ----
    n_logical = 4
    logical_indices = np.arange(n_logical) # Indices in system.logical_basis
    bases = [Wavefunction({"energy": system.logical_basis[:, i]}) for i in logical_indices]

    def simulate_U_q(flux_schedule):
        U = np.zeros((system.n_full, n_logical), dtype=complex)
        for i in range(n_logical):
            evolved_state, _, _, _ = solver.solve(
                system=system,
                initial_state=bases[i].copy(),
                external_voltages=external_voltages,
                coupler_flux_schedule=flux_schedule,
                time=time
            )
            U[:, i] = evolved_state["energy"]
        U_q = np.zeros((n_logical, n_logical), dtype=complex)
        for i in range(n_logical):
            for j in range(n_logical):
                U_q[i, j] = system.logical_basis[:, i].conj().T @ U[:, j]
        return U_q

    def report(U_q, label, U_target):
        print(f"\n========== {label} ==========")
        print("Projected Unitary U_q:")
        print(U_q)
        print("|U_q| (magnitudes):")
        print(np.abs(U_q))
        print("|diag(U_q)|:", np.abs(np.diag(U_q)))
        UdU = U_q.conj().T @ U_q
        print("U_q† U_q (should be ≈ I minus leakage):")
        print(UdU)
        print("max |U_q†U_q - I|:", np.max(np.abs(UdU - np.eye(n_logical))))

        L1 = get_L1(U_q)
        pauli_coefs = get_pauli_coefs(U_q)
        process_fidelity = get_process_fidelity(U_q=U_q, U_target=U_target)
        fidelity = get_average_gate_fidelity(U_q=U_q, process_fidelity=process_fidelity, L1=L1)

        # Conditional phase quotienting out single-qubit Z's.
        # Order: |00>, |01>, |10>, |11>. CZ adds π only to |11>.
        phases = np.angle(np.diag(U_q))
        phi_CZ = phases[3] - phases[1] - phases[2] + phases[0]
        # Wrap to (-pi, pi]
        phi_CZ = (phi_CZ + np.pi) % (2 * np.pi) - np.pi

        print(f"L1 (leakage):       {L1:.6f}")
        print(f"||pauli_coefs||:    {np.linalg.norm(pauli_coefs):.6f}")
        print(f"Process fidelity:   {process_fidelity:.6f}")
        print(f"Avg gate fidelity:  {fidelity:.6f}")
        print(f"Conditional phase φ_CZ / π = {phi_CZ/np.pi:.4f}  (want ±1 for CZ)")
        return fidelity

    # ---- Null test: coupler always OFF -> should yield ≈ identity (up to local Z's) ----
    null_schedule = np.full(num_steps, coupler_flux_off)
    U_q_null = simulate_U_q(null_schedule)
    report(U_q_null, "NULL TEST (coupler OFF, target = I)", np.eye(n_logical, dtype=complex))
    report(U_q_null, "NULL TEST (coupler OFF, target = CZ)", CZ)

    # ---- Main run with the CZ flux pulse ----
    U_q = simulate_U_q(coupler_flux_schedule)
    report(U_q, "CZ RUN", CZ)
    
    
if __name__ == "__main__":
    main()
