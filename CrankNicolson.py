r"""
Crank–Nicolson time evolution of a driven transmon.

Solves the time-dependent Schrödinger equation

.. math::

    i \hbar \, \partial_t |\psi(t)\rangle =
        \big[ H_0 + H_D(t) \big] |\psi(t)\rangle

with the second-order, unconditionally stable Crank–Nicolson scheme.  At each
time step the propagator is approximated by the Cayley form

.. math::

    \Big(\mathbb{1} + \tfrac{i \Delta t}{2 \hbar} H\Big) |\psi_{n+1}\rangle
    = \Big(\mathbb{1} - \tfrac{i \Delta t}{2 \hbar} H\Big) |\psi_n\rangle

evaluated with the Hamiltonian at the *midpoint* of the step (mid-point rule
for the time-dependent drive).

The drive Hamiltonian is

.. math::

    H_D(t) = 2 e \, \frac{C_g}{C_\Sigma} \, V(t) \, \hat n,

i.e. the external voltage couples to the Cooper-pair number operator with an
efficiency set by the coupling-to-total-capacitance ratio.
"""

import numpy as np

from System import System
from Wavefunction import Wavefunction
from constants import e, hbar


class CrankNicolsonSolver():
    """Crank–Nicolson integrator for a driven transmon Hamiltonian."""

    def __init__(self):
        pass

    def solve(self, system: System, initial_state: Wavefunction, external_voltage: np.ndarray, coupler_flux_schedule: np.ndarray, time: np.ndarray):
        r"""Evolve ``initial_state`` under ``H_0 + H_D(t)``.

        Parameters
        ----------
        system : System
            Quantised transmon, providing ``H_0`` and ``\hat n`` in the
            energy basis as well as the circuit's coupling and total
            capacitances.
        initial_state : Wavefunction
            State at ``time[0]`` expressed in the energy basis (the
            ``"energy"`` key must be populated).  This object is mutated in
            place during the loop.
        external_voltage : np.ndarray
            Drive voltage ``V(t)`` sampled on the same grid as ``time``,
            in Volts.
        time : np.ndarray
            Uniformly-spaced time grid in seconds.  The step size is taken
            to be ``time[1] - time[0]``.
        k : int
            The index of the subsystem we want to evolve with Crank Nicolson.

        Returns
        -------
        state : Wavefunction
            Final wavefunction after the last completed step.
        P0, P1, P2 : np.ndarray
            Population (``|c_k|^2``) of the lowest three energy levels at
            each time step.  Shape ``(len(external_voltage),)``.

        Notes
        -----
        The loop runs ``num_steps - 1`` iterations and writes populations
        for indices ``[0, num_steps - 2]``; the final entry of each array
        remains zero by construction.
        """

        n = len(initial_state["energy"])
        num_steps = len(coupler_flux_schedule)
        dt = time[1] - time[0]

        state = initial_state

        P0 = np.zeros(num_steps) # |00>
        P1 = np.zeros(num_steps) # |01>
        P2 = np.zeros(num_steps) # |10>
        
        current_coupler_flux = coupler_flux_schedule[0]
        
        n_vec = np.array([system.upgraded_subsystems[k].n["energy"] for k in range(system.num_subsystems)])

        for i in range(num_steps - 1):
            # ---- Re-Diagonalize if Coupler External Flux Changed ----
            new_coupler_flux = coupler_flux_schedule[i]
    
            if new_coupler_flux != current_coupler_flux:
                system.set_coupler_flux(new_coupler_flux)  # rebuild H0, HC, charge operators
                current_coupler_flux = new_coupler_flux
                
            # ---- Build the Midpoint Driving Hamiltonian ---- 
            HD_midpoint = np.zeros((system.n_full, system.n_full))
            
            n_e_vec = np.zeros(len(system.subsystems))
            
            for k in range(system.num_subsystems):
                v_midpoint = (external_voltage[k][i] + external_voltage[k][i+1]) / 2
                n_e_vec[k] = (system.subsystems[k].circuit.coupling_capacitance * v_midpoint) / (2 * e)     
            
            for k in range(system.num_subsystems):
                for l in range(system.num_subsystems):
                    HD_midpoint += -8 * n_e_vec[k] * system.circuit.charging_energy_matrix[k][l] * n_vec[l]
            
            H = system.H0["energy"] + HD_midpoint

            # Cayley form of the propagator:
            #   A psi_{n+1} = B psi_n,   A = I + i dt/(2 hbar) H,
            #                            B = I - i dt/(2 hbar) H.
            A = np.eye(n) + (1j * (dt / (2*hbar)) * H)
            B = np.eye(n) - (1j * (dt / (2*hbar)) * H)

            state["energy"] = np.linalg.solve(A, B @ state["energy"])

            # Record level populations at the new time.
            P0[i] = np.abs(system.logical_basis[:, 0] @ state["energy"])**2
            P1[i] = np.abs(system.logical_basis[:, 1] @ state["energy"])**2
            P2[i] = np.abs(system.logical_basis[:, 2] @ state["energy"])**2

        return state, P0, P1, P2
