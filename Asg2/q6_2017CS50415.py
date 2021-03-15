import numpy as np

def inner(A, B):
    res = 0
    for i in range(len(A)):
        C = np.conjugate(A[i])*B
        for j in range(len(B)):
            res += C[j]*2/(i+j+1) if (i+j)%2 == 0 else 0
    return res

def orthogonalizePolynomials(P):
    m, n = P.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = P.copy()
    for i in range(n):
        R[i, i] = np.sqrt(inner(V[:, i], V[:, i]))
        Q[:, i] = V[:, i]/R[i, i]
        for j in range(i+1, n):
            R[i, j] = inner(Q[:, i], V[:, j])
            V[:, j] -= R[i, j]*Q[:, i]
    return Q

res = orthogonalizePolynomials(np.eye(5))
for row in res:
    print(' '.join(map(lambda x: "{:.8f}\t".format(x), row)))
