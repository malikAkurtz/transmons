"""
Bundled quantum description of a circuit.

A :class:`System` packages everything the time-evolution stage needs:

* the underlying :class:`~Circuit.Circuit` (used for accessing the
  capacitance matrix and coupling capacitance during driving),
* the **charge operator** ``n``, in both the charge and energy bases, and
* the **bare Hamiltonian** ``H0``, again in both bases.

It is the output of :func:`Quantize.quantize_transmon` and the input to
:class:`CrankNicolson.CrankNicolsonSolver`.
"""

import numpy as np

from Circuit import Circuit
from Operator import Operator
from constants import h

class SubSystem():
    def __init__(self, circuit: Circuit, charge_operator: Operator, unperturbed_hamiltonian: Operator):
        self.circuit = circuit
        self.n       = charge_operator
        self.H0      = unperturbed_hamiltonian
        
class System():
    """Quantised circuit ready for time evolution.

    Parameters
    ----------
    circuit : Circuit
        The lumped-element circuit (typically a :class:`~Transmon.Transmon`).
    charge_operator : Operator
        The dimensionless Cooper-pair number operator ``n`` in the charge
        basis and (after :func:`Quantize.quantize_transmon` runs) in the
        energy eigenbasis.
    unperturbed_hamiltonian : Operator
        The bare (drive-free) Hamiltonian ``H_0``.  Diagonal in the energy
        basis by construction.

    Attributes
    ----------
    state : Wavefunction or None
        Optional slot for the current quantum state during evolution.
        Initialised to ``None`` and left to callers to populate.
    """
    def __init__(self, circuit: Circuit, subsystems: list[SubSystem], n_charge: int, n_trunc: int):
        self.circuit        = circuit
        self.subsystems     = subsystems # unperturbed
        self.num_subsystems = len(self.subsystems)
        self.n_charge       = n_charge
        self.n_trunc        = n_trunc
        self.n_full         = n_trunc**(len(self.subsystems))
        self.state          = None
        
        # --- upgrade subsystems to full tensor product space ---
        self.upgraded_subsystems = [self.upgrade(i, s) for (i,s) in enumerate(self.subsystems)]
            
        # --- coupling Hamiltonian ---
        HC = np.zeros((self.n_full, self.n_full))
        
        for k in range(len(self.subsystems)):
            for l in range(len(self.subsystems)):
                if k == l:
                    continue
                
                HC += 4 * self.upgraded_subsystems[k].n["energy"] * self.circuit.charging_energy_matrix[k][l] \
                    @ self.upgraded_subsystems[l].n["energy"]
                    
        self.HC = Operator(
            basis_to_matrix={"energy": HC}
        )
                    
        # --- joint unperturbed/bare Hamiltonian ---
        H0 = sum(s.H0["energy"] for s in self.upgraded_subsystems) + self.HC["energy"]

        self.H0 = Operator(
            basis_to_matrix={"energy": H0}
        )
        
        # --- diagonalize unperturbed/bare Hamiltonian ---
        # NOTE: if there is only a single transmon, there is no coupling Hamiltonian
        # so self.H0["energy"] is diagonal, and this step is trivial
        # otherwise, HC is non-zero and adds off-diagonal terms
        # to the unperturbed Hamiltonain
        self.energies, self.energy_states = np.linalg.eigh(self.H0["energy"])
        
        # --- construct logical basis ---
        if len(self.subsystems) == 1:
            # eigenstates are the computational states
            # NOTE: the columns are the eigenvectors
            self.logical_basis = self.energy_states[:, :n_trunc]  # first n_trunc eigenstates as columns
            
            E_0 = self.energies[0]
            E_1 = self.energies[1]
            self.frequencies = np.array([(E_1 - E_0) / h])
            self.angular_frequencies = 2 * np.pi * self.frequencies
        elif len(self.subsystems) == 3:
            self.logical_basis = np.zeros((self.n_full, 4))
        
            # bare reference states in the simulation basis
            def bare(i, j, k):
                return np.kron(np.kron(np.eye(self.n_trunc)[i], np.eye(self.n_trunc)[j]), np.eye(self.n_trunc)[k])

            bare_000 = bare(0, 0, 0) # 0
            bare_101 = bare(1, 0, 1) # 3
            bare_100 = bare(1, 0, 0) # 2
            bare_001 = bare(0, 0, 1) # 1

            # |00>_L: ground state, sign-fixed
            psi_0 = self.energy_states[:, 0]
            self.logical_basis[:, 0] = np.sign(bare_000 @ psi_0) * psi_0

            # |11>_L: eigenstate with max overlap on |101>, sign-fixed
            overlaps_101 = bare_101 @ self.energy_states
            idx_11 = np.argmax(np.abs(overlaps_101))
            psi_11 = self.energy_states[:, idx_11]
            self.logical_basis[:, 3] = np.sign(bare_101 @ psi_11) * psi_11

            # |10>_L, |01>_L: Löwdin on the degenerate subspace
            # find the two eigenstates with largest overlap on span{|100>, |001>}
            overlap_100 = bare_100 @ self.energy_states  # shape (n_eigenstates,)
            overlap_001 = bare_001 @ self.energy_states
            total_overlap = overlap_100**2 + overlap_001**2
            # exclude indices already used
            total_overlap[0] = 0.0
            total_overlap[idx_11] = 0.0
            # pick top 2
            degen_indices = np.argsort(total_overlap)[-2:]
            P = self.energy_states[:, degen_indices]  # shape (n_full, 2), columns span the degenerate eigenspace

            # project bare states into eigenspace: |phi_tilde_i> = P P^T |phi_i>
            phi_tilde_1 = P @ (P.T @ bare_100)  # projection of |100> into eigenspace
            phi_tilde_2 = P @ (P.T @ bare_001)  # projection of |001> into eigenspace

            # overlap matrix S_ij = <phi_tilde_i | phi_tilde_j>
            S = np.array([
                [phi_tilde_1 @ phi_tilde_1, phi_tilde_1 @ phi_tilde_2],
                [phi_tilde_2 @ phi_tilde_1, phi_tilde_2 @ phi_tilde_2]
            ])

            # S^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(S)
            S_inv_sqrt = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T

            # Löwdin orthogonalized states: |phi^L_i> = sum_j (S^{-1/2})_{ji} |phi_tilde_j>
            phi_tilde = np.column_stack([phi_tilde_1, phi_tilde_2])  # shape (n_full, 2)
            phi_lowdin = phi_tilde @ S_inv_sqrt  # shape (n_full, 2)

            # |10>_L = phi_lowdin[:, 0], sign-fixed against |100>
            self.logical_basis[:, 2] = np.sign(bare_100 @ phi_lowdin[:, 0]) * phi_lowdin[:, 0]
            # |01>_L = phi_lowdin[:, 1], sign-fixed against |001>
            self.logical_basis[:, 1] = np.sign(bare_001 @ phi_lowdin[:, 1]) * phi_lowdin[:, 1]
            
            # --- dressed Hamiltonian ---
            self.H0["energy_dressed"] = np.diag(self.energies)
            
            # --- extract qubit frequencies from logical basis energies ---
            E_00 = self.logical_basis[:, 0] @ self.H0["energy"] @ self.logical_basis[:, 0]
            E_10 = self.logical_basis[:, 2] @ self.H0["energy"] @ self.logical_basis[:, 2]
            E_01 = self.logical_basis[:, 1] @ self.H0["energy"] @ self.logical_basis[:, 1]
            
            self.frequencies = np.array([
                (E_10 - E_00) / h, # qubit 1 frequency
                (E_01 - E_00) / h  # qubit 2 frequency
            ])
            self.angular_frequencies = 2 * np.pi * self.frequencies
            
        self.periods = 1 / self.frequencies
        
    def upgrade(self, k: int, subsystem: SubSystem):
        n_truncated = subsystem.n.truncate(self.n_trunc)
        n_upgraded  = n_truncated.upgrade(
            k=k,
            num_subsystems=len(self.subsystems),
            dim=self.n_trunc # the dimension of each subsystems Hilbert space
        )
        
        H0_truncated = subsystem.H0.truncate(self.n_trunc)
        H0_upgraded  = H0_truncated.upgrade(
            k=k,
            num_subsystems=len(self.subsystems),
            dim=self.n_trunc
        )
        
        upgraded_subsystem = SubSystem(
            circuit=subsystem.circuit,
            charge_operator=n_upgraded,
            unperturbed_hamiltonian=H0_upgraded
        )
        
        return upgraded_subsystem
    
    def set_coupler_flux(self, new_coupler_flux: float):
        from Quantize import quantize
        # ---- Re-Quantize the Coupler Transmon ----
        new_coupler_subsystem = quantize(
            circuit=self.subsystems[1].circuit,
            charging_energy=self.circuit.charging_energy_matrix[1][1],
            external_flux=new_coupler_flux,
            n_charge=self.n_charge
        )
        
        self.subsystems[1] = new_coupler_subsystem
        
        self.upgraded_subsystems[1] = self.upgrade(
            k=1,
            subsystem=self.subsystems[1]
        )
        
        # --- Re-Compute Coupling Hamiltonian ---
        HC = np.zeros((self.n_full, self.n_full))
        
        for k in range(len(self.subsystems)):
            for l in range(len(self.subsystems)):
                if k == l:
                    continue
                
                HC += 4 * self.upgraded_subsystems[k].n["energy"] * self.circuit.charging_energy_matrix[k][l] \
                    @ self.upgraded_subsystems[l].n["energy"]
                    
        self.HC = Operator(
            basis_to_matrix={"energy": HC}
        )
                    
        # --- Re-Compute Joint Unperturbed/Bare Hamiltonian ---
        H0 = sum(s.H0["energy"] for s in self.upgraded_subsystems) + self.HC["energy"]

        self.H0 = Operator(
            basis_to_matrix={"energy": H0}
        )
        
        # --- Diagonalize New Unperturbed/Bare Hamiltonian ---
        self.energies, self.energy_states = np.linalg.eigh(self.H0["energy"])
        