"""
Scratchpad for hand-checking row-reduction steps of an augmented matrix.

Used during development to verify Gaussian-elimination calculations done
on paper; not part of the simulation pipeline.
"""

import numpy as np


def main():
    """Apply a few row operations to a fixed augmented matrix and print it."""
    A = np.array([
        [1, 0, -1, 0, 3, 1, 156],
        [0, 1, 5/4, 0, 3/4, -1/4, 15],
        [0, 0, -1/2, 1, -1/2, 1/2, 6]
    ], dtype=float)

    # Normalise R2's pivot to 1, then clear it from R1 and R3.
    A[1] = A[1] / (5/4)
    A[0] = A[0] + A[1]
    A[2] = A[2] + (A[1] / 2)

    print(A)


if __name__=="__main__":
    main()
