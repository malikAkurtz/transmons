"""
Physical constants used throughout the simulation.

Re-exports SI constants from :mod:`scipy.constants` and adds the (reduced)
flux quantum, which appears wherever a Josephson element couples to flux.
"""

from scipy.constants import e, h, hbar

# Flux quantum  Phi_0 = h / (2e)  [Wb]; the period of a Josephson cosine
# in flux units.
FLUX_QUANTUM = h / (2 * e)

# Reduced flux quantum  Phi_0^* = hbar / (2e) = Phi_0 / (2 pi)  [Wb].
# Used in the SQUID effective-energy formula and as a natural unit for the
# "external_flux" parameter throughout the codebase.
REDUCED_FLUX_QUANTUM = hbar / (2 * e)
