from affinewarp import AffineWarping
from numpy.testing import assert_allclose, assert_array_equal
import sparse
import numpy as np


def test_dense_transform():

    # data should not be altered by transform before fitting the model.
    model = AffineWarping()
    data = np.random.randn(10, 11, 12)
    for dtype in (np.float64, np.float32, np.int64, np.int32):
        X = data.astype(dtype)
        model.fit(X, iterations=0, verbose=False)
        assert_allclose(X, model.transform(X))


def test_sparse_transform():

    # data should not be altered by transform before fitting the model.
    model = AffineWarping()
    data = sparse.random((10, 11, 12), density=.1)

    for dtype in (np.float64, np.float32, np.int64, np.int32):
        X = data.astype(dtype)
        model.fit(X, iterations=0, verbose=False)
        assert_array_equal(X.coords, model.transform(X).coords)


if __name__ == '__main__':
    test_dense_transform()
    test_sparse_transform()
