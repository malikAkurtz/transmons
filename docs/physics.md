# Physics reference

This file collects the physical models and formulas implemented by the code.

## Units and constants

All quantities are in SI base units (Joules for energies, Farads for
capacitance, Henries for inductance, Volts for voltage, Webers for flux).
The constants in `constants.py`:

| Symbol  | Variable             | Value                |
|---------|----------------------|----------------------|
| `e`     | elementary charge    | `scipy.constants.e`  |
| `h`     | Planck constant      | `scipy.constants.h`  |
| `ℏ`     | `hbar`               | `scipy.constants.hbar` |
| `Φ_0`   | `FLUX_QUANTUM`       | `h / (2e)`           |
| `Φ_0*`  | `REDUCED_FLUX_QUANTUM` | `ℏ / (2e) = Φ_0 / (2π)` |

## Lumped-element circuit Hamiltonian

For a circuit with node-flux coordinates `Φ_i` (one per non-ground node),
the canonical Hamiltonian has the schematic form

```
H = (1/2) Q^T C^{-1} Q  +  (1/2) Φ^T L^{-1} Φ  +  V_J(Φ)
```

where `C` is the capacitance matrix, `L⁻¹` is the inverse-inductance
matrix, `Q` is the conjugate charge of `Φ`, and `V_J(Φ)` is the Josephson
cosine potential.  `Circuit._build_matrices` constructs `C` and `L⁻¹`
following the standard nodal rule:

* off-diagonal `C_{ij} = − Σ` (capacitances of branches between nodes
  `i, j`);
* diagonal `C_{ii} = − Σ_{j ≠ i} C_{ij}` (so rows sum to zero);
* the ground row/column is dropped.

The Josephson potential is added later, in the quantisation step.

## DC SQUID effective Josephson energy

A SQUID is two junctions in parallel threaded by an external flux.  For
junction energies `E_{J,L}, E_{J,R}` and flux `Φ_ext`,

```
                       ┌                                       ┐ 1/2
E_J^eff(Φ_ext) = E_JΣ │ cos²(Φ_ext / 2Φ_0*) + d² sin²(Φ_ext / 2Φ_0*) │
                       └                                       ┘
              = E_JΣ cos(Φ_ext / 2Φ_0*) √( 1 + d² tan²(Φ_ext / 2Φ_0*) )
```

with sum `E_JΣ = E_{J,L} + E_{J,R}` and asymmetry
`d = (E_{J,R} − E_{J,L}) / E_JΣ`.  The second form is what
`DCSQUID.calculate_effective_josephson_energy` evaluates.  Asymmetric
junctions (`d ≠ 0`) keep `E_J^eff` strictly positive, so the qubit
frequency stays finite at every flux bias.

## Transmon Hamiltonian in the charge basis

For a single-island transmon shunted to ground by total capacitance `C_Σ`
and Josephson energy `E_J`, the Hamiltonian is

```
H_0 = 4 E_C n̂²  −  E_J cos(φ̂)
```

with charging energy

```
E_C = e² / (2 C_Σ)
```

and conjugate operators `[φ̂, n̂] = i`.  Working in the discrete charge
basis `{ |n⟩ : n ∈ ℤ }`, the operators read

```
n̂ |n⟩ = n |n⟩,        cos(φ̂) = (1/2) (S + S†),     S |n⟩ = |n + 1⟩.
```

Truncating to `n ∈ {−N, …, +N}` (with `n_charge = 2N + 1` states) gives the
finite-dimensional Hamiltonian assembled in `Quantize.quantize_transmon`:

```
H_0 = 4 E_C diag(n)²  −  (E_J / 2) (S + S†).
```

`np.linalg.eigh` diagonalises this Hermitian matrix; the eigenvectors are
collected as columns of a unitary `U`, and any operator `O` is rotated to
the energy basis via `U† O U`.

The lowest three eigenvalues give

```
f_01 = (E_1 − E_0) / h
α    = (E_2 − E_1) − (E_1 − E_0)        (anharmonicity).
```

A typical transmon has `E_J / E_C ≈ 50` and `α / h` of order
`−E_C / h ≈ −200 MHz`.

## Drive Hamiltonian

The transmon couples to an external voltage `V(t)` through the coupling
capacitance `C_g`.  The drive Hamiltonian is

```
H_D(t) = 2e (C_g / C_Σ) V(t) n̂.
```

The factor `C_g / C_Σ` is the capacitive division between the drive line
and the qubit island.  Implemented in `CrankNicolson.CrankNicolsonSolver.solve`.

## Crank–Nicolson propagator

The time-dependent Schrödinger equation

```
iℏ ∂_t |ψ⟩ = H(t) |ψ⟩
```

is integrated by the Crank–Nicolson (Cayley) scheme

```
( I + i Δt / (2ℏ) H_{n+1/2} ) ψ_{n+1}  =  ( I − i Δt / (2ℏ) H_{n+1/2} ) ψ_n
```

where `H_{n+1/2}` uses the midpoint voltage
`V_{n+1/2} = ½(V_n + V_{n+1})`.  This scheme is unconditionally stable,
unitary at second order in `Δt`, and is exactly what `solve()` does each
step via `numpy.linalg.solve`.

## SFQ pulse waveforms

Two pulse-train generators are provided.

### Gaussian train (`utils.create_gaussian_sfq_pulses`)

```
V(t) = A  Σ_{k=0}^{N-1}  exp( −(t − k T_d)² / (2σ)² )
```

The amplitude in `config.py` is set to

```
A = Φ_0* / ( σ √(2π) )
```

so that the time-integral of one Gaussian voltage spike equals exactly the
reduced flux quantum — i.e. each pulse delivers one SFQ "kick".

### RCSJ-derived realistic waveform (`riyas_pulses.py`)

The Resistively and Capacitively Shunted Junction equation

```
C dV/dt + V/R + I_c sin(φ) = I(t),       V = (ℏ / 2e) dφ/dt
```

is integrated with `scipy.integrate.solve_ivp` for a junction biased near
`I_c` and pulsed by rectangular current excursions of width `rect_width`.
A SciPy *event* terminates each segment after the phase has slipped by
exactly `2π`, ensuring each pulse delivers one flux quantum
`Φ_0 = h / (2e)`.  The script downsamples the resulting voltage trace to a
100k-point uniform grid and writes it to `sfq_V_lookup.csv`.
