"""Smoke tests for multistore computing module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_calcium_store_init():
    from multistore_computing.store import CalciumStore
    store = CalciumStore("Test", size=50, speed=0.5, capacity=1.0)
    assert store.name == "Test"
    assert store.size == 50
    assert store.activation.shape == (50,)
    assert store.contents.shape == (50, 3)


def test_calcium_store_write_read():
    from multistore_computing.store import CalciumStore
    np.random.seed(42)
    store = CalciumStore("Test", size=50, speed=0.5, capacity=1.0, reliability=1.0)
    success = store.write(0, [0.5, 0.3, 0.1])
    assert success
    data = store.read(0)
    assert data is not None
    np.testing.assert_array_almost_equal(data, [0.5, 0.3, 0.1])


def test_multi_store_computer_init():
    from multistore_computing.computer import MultiStoreComputer
    computer = MultiStoreComputer()
    assert 'primary' in computer.stores
    assert 'secondary' in computer.stores
    assert 'tertiary' in computer.stores


def test_multi_store_computer_write_query():
    from multistore_computing.computer import MultiStoreComputer
    np.random.seed(42)
    computer = MultiStoreComputer()
    success, store_name = computer.write_data(0.5, priority='high')
    assert success
    assert store_name == 'primary'
    computer.run_simulation(steps=5)
    results = computer.query(0.5)
    assert isinstance(results, list)


def test_multi_store_computer_update():
    from multistore_computing.computer import MultiStoreComputer
    np.random.seed(42)
    computer = MultiStoreComputer()
    computer.update()
    assert computer.time_step == 1
    assert 0 in computer.store_activations


if __name__ == "__main__":
    test_calcium_store_init()
    test_calcium_store_write_read()
    test_multi_store_computer_init()
    test_multi_store_computer_write_query()
    test_multi_store_computer_update()
    print("All multistore computing tests passed.")
