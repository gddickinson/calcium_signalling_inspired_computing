"""Wave logic gates read their two input patterns (bit = pattern mean > 0.5) and follow the truth tables."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_based_computing import WaveComputing, WaveComputingMedium  # noqa: E402

TABLES = {'and': [0, 0, 0, 1], 'or': [0, 1, 1, 1], 'xor': [0, 1, 1, 0]}
BITS = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.fixture(scope='module')
def computer():
    return WaveComputing(WaveComputingMedium(size=(60, 60)))


@pytest.mark.parametrize('op', sorted(TABLES))
def test_truth_table(computer, op):
    outs = []
    for a, b in BITS:
        r = computer.wave_logic_gates([np.full((4, 4), float(a)), np.full((4, 4), float(b))], operation=op)
        assert r['inputs'] == [bool(a), bool(b)]
        outs.append(int(r['output']))
    assert outs == TABLES[op], f"{op}: {outs}"


def test_gate_readout_margins(computer):
    """Readouts sit well away from the thresholds (not a knife-edge result)."""
    one = computer.wave_logic_gates([np.ones((2, 2)), np.zeros((2, 2))], operation='or')
    both = computer.wave_logic_gates([np.ones((2, 2)), np.ones((2, 2))], operation='and')
    xor = computer.wave_logic_gates([np.ones((2, 2)), np.ones((2, 2))], operation='xor')
    ref = one['reference']
    assert ref > 0.05
    assert both['result_value'] > 1.7 * ref
    assert xor['result_value'] < 0.1 * ref


def test_needs_two_inputs(computer):
    assert computer.wave_logic_gates([np.ones((2, 2))]) is None
