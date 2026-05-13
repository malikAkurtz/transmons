
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import e, h, REDUCED_FLUX_QUANTUM
from config import NUM_KICKS, AMPLITUDE_SCALE, DETUNING, STEPS_PER_PERIOD, SIGMA
from Node import Node
from System import System
from DCSQUID import DCSQUID
from Transmon import Transmon
from Quantize import quantize
from CrankNicolson import CrankNicolsonSolver
from Wavefunction import Wavefunction
from Branch import Capacitor
from utils import create_gaussian_sfq_pulses
from Circuit import Circuit
from Graph import Multidigraph


def main():
    """Run the end-to-end transmon-driving simulation."""
    # ------------------------ Hyperparameters ------------------------
    # Charge basis size; states run from -(n-1)/2 to +(n-1)/2.  Must be
    # odd so the basis is symmetric around zero.
    n_charge = 101
    n_trunc  = 5

    # ---- Multi-Qubit Circuit Paramters ----
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
    left_jj_energy_c       = 18e-9  * REDUCED_FLUX_QUANTUM  # [J]
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
    
    # ---- Create Graph Representation ----
    graph = Multidigraph(
        nodes=[gnd, transmon_1.island, transmon_c.island, transmon_2.island],
        branches=transmon_1.branches + transmon_c.branches + transmon_2.branches + [cap_12, cap_1c, cap_2c],
        s=transmon_1.s + transmon_c.s + transmon_2.s + [transmon_1.island._id, transmon_1.island._id, transmon_2.island._id],
        t=transmon_1.t + transmon_c.t + transmon_2.t + [transmon_2.island._id, transmon_c.island._id, transmon_c.island._id],
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

    # ---- Qubit Subsystem Index to Drive ----
    k = 0
    
    # ---- Initialize State ----
    # Start in the ground state of the energy basis: amplitude 1 on |0>.
    prob_amps = np.zeros(n_trunc**len(subsystems))
    prob_amps[0] = 1
    initial_state = Wavefunction(basis_to_coefs={"energy": prob_amps})
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

    # ---- Evolve ----
    final_state, P0, P1, P2 = solver.solve(
        system=system,
        initial_state=initial_state,
        external_voltage=external_voltage,
        time=time,
        k=k
    )

    # ---- Plot ground / first / second level populations vs time ----
    plt.plot(time, P0, label='P0')
    plt.plot(time, P1, label='P1')
    plt.plot(time, P2, label='P2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
