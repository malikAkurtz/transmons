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


def main():
    """Run the end-to-end transmon-driving simulation."""
    # ------------------------ Hyperparameters ------------------------
    # Charge basis size; states run from -(n-1)/2 to +(n-1)/2.  Must be
    # odd so the basis is symmetric around zero.
    n_charge = 101
    n_trunc  = n_charge

    # SQUID + drive parameters.
    external_flux        = 0.130 * REDUCED_FLUX_QUANTUM
    shunt_capacitance    = 70e-15   # [F]
    coupling_capacitance = 1e-15    # [F]

    # Junction parameters.  Asymmetric junctions give a flux-tunable but
    # finite minimum E_J.
    left_jj_capacitance    = 0
    left_josephson_energy  = 7e-9  * REDUCED_FLUX_QUANTUM  # [J]
    right_jj_capacitance   = 0
    right_josephson_energy = 21e-9 * REDUCED_FLUX_QUANTUM  # [J]

    # ---- Build Ground Node ----
    gnd = Node()

    # ---- Build DCSQUID Circuit Object ----
    dcsquid = DCSQUID(
        ground_node=gnd,
        left_jj_capacitance=left_jj_capacitance,
        left_josephson_energy=left_josephson_energy,
        right_jj_capacitance=right_jj_capacitance,
        right_josephson_energy=right_josephson_energy
    )
    
    # ---- Build Transmon Circuit Object ----
    transmon = Transmon(
        dcsquid=dcsquid,
        shunt_capacitance=shunt_capacitance,
        coupling_capacitance=coupling_capacitance
    )

    # ---- Partition Circuit Nodes and Build Capacitance/Inverse Inductance Matrices
    transmon.build()
    
    print("Capacitance Matrix [fF]:")
    print(transmon.capacitance_matrix * 1e15)
    
    # ---- Retrieve Transmon Charging Energy ----
    EC = transmon.charging_energy
    
    # ---- Retrieve Transmon Josephson Energy (From DCSQUID) ----
    EJ = DCSQUID.calculate_effective_josephson_energy(
        left_josephson_energy=left_josephson_energy,
        right_josephson_energy=right_josephson_energy,
        external_flux=external_flux
    )

    print(f"EC = {EC}  EJ = {EJ}  EJ/EC = {EJ/EC}")

    # ---- Quantize ----
    transmon_subsystem = quantize(
        circuit=transmon, 
        external_flux=external_flux, 
        n_charge=n_charge,
    )
    
    system = System(
        circuit=transmon,
        subsystems=[transmon_subsystem],
        n_charge=n_charge,
        n_trunc=n_trunc
    )
    
    print("Idling Hamiltonian: ")
    print(system.hamiltonian(external_flux))
    
    # ---- Qubit Frequency ----
    f_01 = (transmon_subsystem.H0["energy"][1][1] - transmon_subsystem.H0["energy"][0][0]) / h
    
    # ---- Qubit Angular Frequency ----
    omega_01 = 2 * np.pi * f_01
    
    # ---- System Anharmonicity ----
    alpha    = (transmon_subsystem.H0["energy"][2][2] - transmon_subsystem.H0["energy"][1][1]) \
             - (transmon_subsystem.H0["energy"][1][1] - transmon_subsystem.H0["energy"][0][0])
    print(f"f_01 = {f_01/1e9:.4f} GHz  |  alpha = {alpha/h/1e6:.2f} MHz")

    # ---- Drive ----
    # Start in the ground state of the energy basis: amplitude 1 on |0>.
    initial_state = Wavefunction(basis_to_coefs={"energy": np.array([1] + [0] * (n_charge - 1))})
    solver        = CrankNicolsonSolver()

    # To use the SFQ lookup table instead of Gaussians, uncomment:
    # lookup = pd.read_csv("sfq_V_lookup.csv")
    # time, external_voltage = lookup["time_s"].to_numpy(), lookup["voltage_V"].to_numpy()

    time, external_voltage = create_gaussian_sfq_pulses(
        num_kicks=NUM_KICKS,
        amplitude_scale=AMPLITUDE_SCALE,
        driving_period=(1 / f_01) + DETUNING,
        pulse_width=SIGMA,
        steps_per_period=STEPS_PER_PERIOD
    )

    # ---- Evolve ----
    final_state, P0, P1, P2 = solver.solve(
        system=system,
        initial_state=initial_state,
        external_voltage=external_voltage,
        time=time,
        k=0
    )

    # ---- Plot ground / first / second level populations vs time ----
    plt.plot(time, P0, label='P0')
    plt.plot(time, P1, label='P1')
    plt.plot(time, P2, label='P2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
