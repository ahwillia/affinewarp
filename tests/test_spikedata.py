import numpy as np
from affinewarp.spikedata import _sequentially_renumber, is_sorted


def test_is_sorted():
    np.random.seed(1234)

    for _ in range(100):
        foo = np.random.randn(100)
        foo.sort()
        assert is_sorted(foo)
        assert not is_sorted(foo[::-1])
        foo[0] = foo[-1]
        assert not is_sorted(foo)


def test_seq_renumbering():
    np.random.seed(1234)

    foo = np.array([1, 2, 3, 4])
    _sequentially_renumber(foo)
    for i in range(len(foo)):
        assert foo[i] == i

    for _ in range(100):
        foo = np.random.randint(10, size=100)
        foo.sort()
        _sequentially_renumber(foo)
        assert is_sorted(foo)
        assert all(np.diff(foo) < 2)
        assert all(np.diff(foo) > -1)

    for _ in range(100):
        foo = np.random.randint(1000, size=100)
        foo.sort()
        _sequentially_renumber(foo)
        assert is_sorted(foo)
        assert all(np.diff(foo) < 2)
        assert all(np.diff(foo) > -1)
