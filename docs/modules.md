# Module reference

A short, alphabetised reference to every module.  Each entry summarises the
public surface; full docstrings live in the source files.

## `Branch.py`

Two-terminal lumped element primitives.

* `Branch` — abstract base; holds the shared `global_id` counter.
* `Capacitor(capacitance)` — linear capacitor `[F]`.
* `Inductor(inductance)` — linear inductor `[H]`.
* `JosephsonElement(josephson_energy)` — Josephson element `E_J [J]`.
  Geometric junction capacitance is represented as a sibling `Capacitor`.

## `Circuit.py`

Generic circuit assembled from a `Multidigraph`.

* `Circuit(graph)` — stores topology; matrices are `None` until `build()` runs.
* `Circuit.build()` — partitions nodes (`active_nodes`, `passive_nodes`) and
  builds the reduced `capacitance_matrix` and `inv_inductance_matrix`.
* `str(circuit)` — readable list of branches with element type and
  source→target ids.

## `config.py`

Default Gaussian-SFQ drive parameters consumed by `main.py`.

* `NUM_KICKS`, `SIGMA`, `AMPLITUDE_SCALE`, `DETUNING`, `STEPS_PER_PERIOD`.
* `AMPLITUDE_SCALE` is set so that `∫ V(t) dt` per pulse equals the reduced
  flux quantum `ℏ / (2e)`.

## `constants.py`

Re-exports `e`, `h`, `hbar` from `scipy.constants` and adds:

* `FLUX_QUANTUM = h / (2e)`
* `REDUCED_FLUX_QUANTUM = ℏ / (2e)`

## `CrankNicolson.py`

* `CrankNicolsonSolver()` — no-arg constructor.
* `solver.solve(system, initial_state, external_voltage, time)` — Cayley-form
  Crank–Nicolson integrator.  Returns `(state, P0, P1, P2)` with level
  populations along `time`.

## `DCSQUID.py`

* `DCSQUID(ground_node, left_jj_capacitance, left_josephson_energy,
   right_jj_capacitance, right_josephson_energy)` — builds a two-junction
  SQUID.  Subclass of `Circuit`; exposes `self.island`.
* `DCSQUID.calculate_effective_josephson_energy(E_JL, E_JR, Φ_ext)` —
  static helper for the flux-tunable effective Josephson energy.

## `Graph.py`

* `Multidigraph(nodes, branches, s, t)` — bundle of the four arrays needed
  to describe a directed multigraph topology.

## `main.py`

End-to-end demo entry point.  Runs the build → quantise → drive → plot
pipeline with a flux-biased asymmetric SQUID transmon and a 150-pulse
Gaussian SFQ train.

## `Node.py`

* `Node()` — auto-incrementing integer id (`_id`).  The first instance
  created (id `0`) is treated as ground.

## `Operator.py`

* `Operator(basis_to_matrix)` — multi-basis operator container.
* `op[basis]` getter and `op[basis] = matrix` setter for adding new basis
  representations.

## `Quantize.py`

* `quantize_transmon(transmon, external_flux, n_charge)` — builds the
  charge-basis transmon Hamiltonian, diagonalises it, and returns a
  `System` with `H_0` and `n̂` populated in both `"charge"` and `"energy"`
  bases.

## `riyas_pulses.py`

Standalone script.  Integrates the RCSJ equation for a current-pulsed
Josephson junction, terminating each pulse on a `2π` phase slip, and writes
the resulting SFQ voltage waveform (with current) to `sfq_V_lookup.csv`.

## `scripts/`

* `scripts/main1.py` — placeholder/skeleton script (incomplete).
* `scripts/playground.py` — scratch row-reduction sanity-check.

## `System.py`

* `System(circuit, charge_operator, unperturbed_hamiltonian)` — bundles a
  built circuit with its quantised charge operator `n` and Hamiltonian
  `H_0`.  Has a `state` slot for callers to populate during evolution.

## `Transmon.py`

* `Transmon(dcsquid, shunt_capacitance, coupling_capacitance)` — wraps a
  `DCSQUID` with the additional shunt and coupling capacitors.  Subclass
  of `Circuit`.

## `utils.py`

* `create_gaussian_sfq_pulses(num_kicks, amplitude_scale, driving_period,
   pulse_width, steps_per_period)` — synthesise the analytic Gaussian
  pulse train; returns `(time, voltage)`.

## `Wavefunction.py`

* `Wavefunction(basis_to_coefs)` — multi-basis state container.
* Tracks a cumulative unitary `U` per basis (initialised to identity).
* `wavefunction.apply(operator)` — apply an `Operator` in every shared
  basis.
