import numpy as np
from affinewarp.tridiag import trisolve
import itertools

atol_float64 = 1e-8


def test_trisolver():
    np.random.seed(1234)
    A = np.diag(np.random.randn(100)) + np.diag(np.random.randn(99), 1)
    A = np.dot(A.T, A)

    a, b, c = np.diag(A, -1), np.diag(A), np.diag(A, 1)

    d = np.random.randn(100, 10)

    x1 = trisolve(a, b, c, d)
    x2 = trisolve(a, b, a, d)
    x3 = np.linalg.solve(A, d)

    for u, v in itertools.combinations((x1, x2, x3), 2):
        print('woooo')
        assert np.linalg.norm(x1 - x2) < atol_float64
