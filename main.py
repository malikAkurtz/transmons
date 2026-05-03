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
from Quantize import quantize
from CrankNicolson import CrankNicolsonSolver
from Wavefunction import Wavefunction
from utils import create_gaussian_sfq_pulses


def main():
    """Run the end-to-end transmon-driving simulation."""
    # ------------------------ Hyperparameters ------------------------
    # Charge basis size; states run from -(n-1)/2 to +(n-1)/2.  Must be
    # odd so the basis is symmetric around zero.
    n_charge = 201

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

    # ---- Build circuit ----
    gnd = Node()

    dcsquid = DCSQUID(
        ground_node=gnd,
        left_jj_capacitance=left_jj_capacitance,
        left_josephson_energy=left_josephson_energy,
        right_jj_capacitance=right_jj_capacitance,
        right_josephson_energy=right_josephson_energy
    )

    transmon = Transmon(
        dcsquid=dcsquid,
        shunt_capacitance=shunt_capacitance,
        coupling_capacitance=coupling_capacitance
    )
    print("Transmon Circuit Representation")
    print(transmon)

    transmon.build()
    print("Capacitance Matrix [fF]:")
    print(transmon.capacitance_matrix * 1e15)

    # Charging energy E_C and effective Josephson energy E_J for the
    # current flux bias; the ratio sets the transmon regime.
    EC = e**2 / (2 * transmon.capacitance_matrix[0][0])
    EJ = DCSQUID.calculate_effective_josephson_energy(
        left_josephson_energy=left_josephson_energy,
        right_josephson_energy=right_josephson_energy,
        external_flux=external_flux
    )
    print(f"EC = {EC}  EJ = {EJ}  EJ/EC = {EJ/EC}")
    
    print("Hamiltonian: ")
    print(transmon.hamiltonian(external_flux))

    # ---- Quantize ----
    system = quantize(
        circuit=transmon, 
        external_flux=external_flux, 
        n_charge=n_charge
    )
    
    # Qubit transition frequency f_01 and anharmonicity alpha extracted
    # from the lowest three energy eigenvalues.
    f_01     = (system.H0["energy"][1][1] - system.H0["energy"][0][0]) / h
    omega_01 = 2 * np.pi * f_01
    alpha    = (system.H0["energy"][2][2] - system.H0["energy"][1][1]) \
             - (system.H0["energy"][1][1] - system.H0["energy"][0][0])
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
        time=time
    )

    # ---- Plot ground / first / second level populations vs time ----
    plt.plot(time, P0, label='P0')
    plt.plot(time, P1, label='P1')
    plt.plot(time, P2, label='P2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
