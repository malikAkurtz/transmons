import numpy as np

def main():
    A = np.array([
        [1, 0, -1, 0, 3, 1, 156],
        [0, 1, 5/4, 0, 3/4, -1/4, 15],
        [0, 0, -1/2, 1, -1/2, 1/2, 6]
    ], dtype=float)

    A[1] = A[1] / (5/4)
    A[0] = A[0] + A[1]
    A[2] = A[2] + (A[1] / 2)

    print(A)

if __name__=="__main__":
    main()
