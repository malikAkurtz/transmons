import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import e, h, REDUCED_FLUX_QUANTUM
from config import NUM_KICKS, AMPLITUDE_SCALE, DETUNING, STEPS_PER_PERIOD, SIGMA
from Node import Node
from DCSQUID import DCSQUID
from Transmon import Transmon
from Quantize import quantize_transmon
from CrankNicolson import CrankNicolsonSolver
from Wavefunction import Wavefunction
from utils import create_gaussian_sfq_pulses


def main():
    # ------------------------ Hyperparameters ------------------------
    n_charge = 201  # charge states from -(n-1)/2 to (n-1)/2; must be odd

    external_flux        = 0.130 * REDUCED_FLUX_QUANTUM
    shunt_capacitance    = 70e-15   # [F]
    coupling_capacitance = 1e-15    # [F]

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

    EC = e**2 / (2 * transmon.capacitance_matrix[0][0])
    EJ = DCSQUID.calculate_effective_josephson_energy(
        left_josephson_energy=left_josephson_energy,
        right_josephson_energy=right_josephson_energy,
        external_flux=external_flux
    )
    print(f"EC = {EC}  EJ = {EJ}  EJ/EC = {EJ/EC}")

    # ---- Quantize ----
    system = quantize_transmon(transmon=transmon, external_flux=external_flux, n_charge=n_charge)

    f_01     = (system.H0["energy"][1][1] - system.H0["energy"][0][0]) / h
    omega_01 = 2 * np.pi * f_01
    alpha    = (system.H0["energy"][2][2] - system.H0["energy"][1][1]) \
             - (system.H0["energy"][1][1] - system.H0["energy"][0][0])
    print(f"f_01 = {f_01/1e9:.4f} GHz  |  alpha = {alpha/h/1e6:.2f} MHz")

    # ---- Drive ----
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

    plt.plot(time, P0, label='P0')
    plt.plot(time, P1, label='P1')
    plt.plot(time, P2, label='P2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
