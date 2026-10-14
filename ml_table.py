"""
Reproduces Table "Criss-cross hierarchy" of the multilevel section.

Criss-cross hierarchy on (0,1)^2 with s_l = 2^-l, l = 0..L. Unknowns of level l:
interior nodes and all cell centres (I_Omega^l). For the finest level L:
  P        Gram (stiffness) matrix of the raw weights: 4 on the diagonal, -1 node-cell
  I_l      level-l weights expressed in the finest weights (values at finest unknowns)
  B        sum_l I_l D_l^{-1} I_l^T,   D_l = diag(I_l^T P I_l)   (eta_ML^2 = r^T B r)
Reports lambda_min, lambda_max of B P, K_ML = lambda_min^{-1/2}, and the single-level
constant lambda_min(D_L^{-1} P)^{-1/2}.

Usage: python ml_table.py [--Lmax 7]
"""
import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def unknowns(n):
    s = 1 / n
    nodes = [(i * s, j * s, "node") for i in range(1, n) for j in range(1, n)]
    cells = [((i + .5) * s, (j + .5) * s, "cell") for i in range(n) for j in range(n)]
    return nodes + cells


def hat(n, v, X, Y):
    s = 1 / n
    if v[2] == "node":
        return np.clip(1 - (abs(X - v[0]) + abs(Y - v[1])) / s, 0, None)
    return np.clip(1 - 2 * np.maximum(abs(X - v[0]), abs(Y - v[1])) / s, 0, None)


def gram(n):
    V = unknowns(n)
    key = {(round(v[0] * 2 * n), round(v[1] * 2 * n), v[2]): k for k, v in enumerate(V)}
    rows, cols, vals = list(range(len(V))), list(range(len(V))), [4.0] * len(V)
    for k, v in enumerate(V):
        if v[2] == "cell":
            for a in (-1, 1):
                for b in (-1, 1):
                    j = key.get((round(v[0] * 2 * n) + a, round(v[1] * 2 * n) + b, "node"))
                    if j is not None:
                        rows += [k, j]; cols += [j, k]; vals += [-1.0, -1.0]
    return sp.csc_matrix((vals, (rows, cols)), shape=(len(V), len(V)))


def prolongation(lc, L):
    Vf = unknowns(2 ** L)
    X = np.array([v[0] for v in Vf]); Y = np.array([v[1] for v in Vf])
    rows, cols, vals = [], [], []
    for k, v in enumerate(unknowns(2 ** lc)):
        h = hat(2 ** lc, v, X, Y)
        nz = np.nonzero(h > 1e-14)[0]
        rows += list(nz); cols += [k] * len(nz); vals += list(h[nz])
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(Vf), len(unknowns(2 ** lc))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lmax", type=int, default=7)
    Lmax = ap.parse_args().Lmax
    print(" L    n   dofs  lam_min(BP)  lam_max(BP)   K_ML   single-level")
    for L in range(1, Lmax + 1):
        P = gram(2 ** L)
        n = P.shape[0]
        Is = [prolongation(l, L) for l in range(L + 1)]
        Ds = [1 / (I.T @ P @ I).diagonal() for I in Is]
        B = lambda x: sum(I @ (d * (I.T @ x)) for I, d in zip(Is, Ds))
        # B P is self-adjoint in the P inner product: solve P B P y = lam P y
        lu = spl.splu(P)
        A = spl.LinearOperator((n, n), matvec=lambda x: P @ B(P @ x), dtype=float)
        Minv = spl.LinearOperator((n, n), matvec=lu.solve, dtype=float)
        if n < 400:
            Bm = np.column_stack([B(e) for e in np.eye(n)])
            lam = np.linalg.eigvals(Bm @ P.toarray()).real
            lo, hi = lam.min(), lam.max()
        else:
            hi = spl.eigsh(A, k=1, M=P, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-9)[0]
            lo = spl.eigsh(A, k=1, M=P, Minv=Minv, which="SA", return_eigenvectors=False, tol=1e-9,
                           maxiter=50000)[0]
        single = spl.eigsh(P / 4.0, k=1, sigma=0, which="LM", return_eigenvectors=False)[0]
        print(f"{L:2d} {2**L:4d} {n:6d}   {lo:10.3f}   {hi:10.3f}   {lo**-0.5:6.3f}   {single**-0.5:8.2f}")


if __name__ == "__main__":
    main()
