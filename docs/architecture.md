# Architecture

This document explains how the modules fit together and how data flows from
a circuit description to a populated time-evolution result.

## Layered design

The code is organised as five thin layers.  Each layer depends only on the
layers below it.

```
┌─────────────────────────────────────────────────────────────┐
│  5.  Entry point             main.py                        │
├─────────────────────────────────────────────────────────────┤
│  4.  Dynamics                CrankNicolson.py               │
├─────────────────────────────────────────────────────────────┤
│  3.  Quantisation            Quantize.py, System.py         │
│                              Operator.py, Wavefunction.py   │
├─────────────────────────────────────────────────────────────┤
│  2.  Specific circuits       Transmon.py, DCSQUID.py        │
├─────────────────────────────────────────────────────────────┤
│  1.  Topology + matrices     Node.py, Branch.py, Graph.py,  │
│                              Circuit.py                     │
├─────────────────────────────────────────────────────────────┤
│  0.  Constants & helpers     constants.py, config.py,       │
│                              utils.py, riyas_pulses.py      │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1 — Topology and matrices

`Node` hands out unique integer ids; the first one created (id `0`) is
treated as ground throughout the pipeline.

`Branch` is the abstract base for two-terminal elements.  Three concrete
elements exist: `Capacitor`, `Inductor`, `JosephsonElement`.  Every branch
also gets an auto-incrementing id, which is used as an index into the
source/target arrays.

`Multidigraph` is a tiny dataclass-like bundle: nodes, branches, and the
parallel `s` and `t` arrays giving each branch's source-node id and
target-node id.

`Circuit` consumes a `Multidigraph` and is the workhorse.  Calling
`circuit.build()`:

1. Partitions every non-ground node into *active* (touches at least one
   inductive element) and *passive* (only capacitors).
2. Constructs the full `(P × P)` capacitance matrix `C` and inverse-
   inductance matrix `L⁻¹` via the standard nodal rule:
   off-diagonal `[i, j]` is `−sum(branch values)` over branches between
   nodes `i` and `j`; diagonal `[i, i]` is set so each row sums to zero.
3. Drops the ground row/column to give the *reduced* matrices used
   downstream.

Josephson cosine non-linearity is **not** present in the matrices; it is
applied later, during quantisation.

## Layer 2 — Specific circuits

`DCSQUID` builds the two-junction SQUID:

* Creates one **island** node and uses the supplied `ground_node` for
  ground.
* Adds four branches: two `JosephsonElement`s (one per junction) and two
  matching `Capacitor`s for their geometric capacitance.
* Forwards the assembled topology to `Circuit`.

`DCSQUID.calculate_effective_josephson_energy` is a pure static helper that
returns the flux-tunable effective `E_J(Φ_ext)` of the SQUID.

`Transmon` wraps a pre-built `DCSQUID` and adds two more capacitors:

* a **shunt** capacitor that pushes `E_J / E_C` into the transmon regime,
* a **coupling** capacitor that the external drive enters through.

The combined topology is passed to `Circuit`, and `transmon.build()`
populates `transmon.capacitance_matrix`.

## Layer 3 — Quantisation

`Operator` and `Wavefunction` are dual containers: each stores a
representation in *every* basis it has been computed for, keyed by a
string label such as `"charge"` or `"energy"`.  This avoids recomputing
basis transformations and keeps callsites readable: `H["charge"]`,
`H["energy"]`.

`quantize_transmon(transmon, external_flux, n_charge)`:

1. Reads the total island capacitance from `transmon.capacitance_matrix`.
2. Computes `E_C = e² / (2 C_Σ)`.
3. Computes `E_J(Φ_ext)` via the SQUID helper.
4. Builds the symmetric integer charge basis `n ∈ {-N, …, +N}` with
   `N = (n_charge − 1) / 2`.
5. Constructs the charge-basis Hamiltonian `H_0 = 4 E_C n̂² − (E_J/2)(S + S†)`.
6. Diagonalises `H_0` and stores `H_0` and the charge operator in the
   energy basis as well.
7. Returns a `System(circuit, n, H_0)`.

## Layer 4 — Dynamics

`CrankNicolsonSolver.solve(system, initial_state, external_voltage, time)`
integrates `i ℏ ∂_t |ψ⟩ = (H_0 + H_D(t)) |ψ⟩` on a uniform grid:

* The drive Hamiltonian is
  `H_D(t) = 2e (C_g / C_Σ) V(t) n̂` in the energy basis.
* Each step evaluates `H` at the *midpoint* voltage and applies the Cayley
  form
  `(I + iΔt/(2ℏ) H) ψ_{n+1} = (I − iΔt/(2ℏ) H) ψ_n`,
  solving the linear system once per step with `numpy.linalg.solve`.
* Populations of `|0⟩, |1⟩, |2⟩` are recorded into `P0, P1, P2` arrays.

## Layer 5 — Entry point

`main.py` chains everything together and shows the result:

1. Build a `Node` for ground, then a `DCSQUID` and a `Transmon`.
2. `transmon.build()` to populate the capacitance matrix.
3. Print `E_C`, `E_J`, and the ratio.
4. `quantize_transmon(...)` to get the `System`.
5. Compute `f_01` and `α` from the lowest three eigenvalues.
6. Build the initial wavefunction (ground state in the energy basis).
7. Synthesise the drive waveform with `create_gaussian_sfq_pulses` (or
   load `sfq_V_lookup.csv`).
8. `solver.solve(...)` and `matplotlib.pyplot` the populations.

## Drive waveform sources

Two interchangeable drive generators are provided:

* `utils.create_gaussian_sfq_pulses` — analytic Gaussian train, parameters
  in `config.py`.  Used by default in `main.py`.
* `riyas_pulses.py` — full RCSJ integration of a biased Josephson junction
  pulsed by a rectangular current train, written to
  `sfq_V_lookup.csv` for fast loading.  More physically realistic.

`main.py` contains a commented example showing how to swap one for the
other.
